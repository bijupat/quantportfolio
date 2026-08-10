import os
import json
from pathlib import Path
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError
from market_data.models import TrainedModel
from core.models import Universe

class Command(BaseCommand):
    help = "Imports legacy .h5 model weights and JSON metadata into the database."

    def add_arguments(self, parser):
        parser.add_argument('--name', type=str, required=True, help='Base name of the model (e.g., transformer_nifty50)')
        parser.add_argument('--models-dir', type=str, default='models', help='Directory containing the .h5 files')
        parser.add_argument('--metrics-dir', type=str, default='metrics', help='Directory containing the JSON metadata')
        parser.add_argument('--universe', type=str, default='nifty50', help='Slug of the Universe this model was trained on')

    def handle(self, *args, **options):
        model_name = options['name']
        models_dir = Path(options['models_dir'])
        metrics_dir = Path(options['metrics_dir'])
        universe_slug = options['universe']

        self.stdout.write(self.style.MIGRATE_HEADING(f"Starting import for legacy model: {model_name}"))

        # 1. Verify all files exist
        weights_path = models_dir / f"{model_name}.weights.h5"
        arch_path = metrics_dir / f"{model_name}_arch.json"
        scalers_path = metrics_dir / f"{model_name}_scalers.json"
        eval_path = metrics_dir / f"{model_name}_eval.json"

        missing_files = []
        for p in [weights_path, arch_path, scalers_path]:
            if not p.exists():
                missing_files.append(str(p))
        
        if missing_files:
            raise CommandError(f"Cannot import model. Missing critical files:\n" + "\n".join(missing_files))

        # 2. Load JSON metadata
        try:
            with open(arch_path, 'r') as f:
                arch_data = json.load(f)
            with open(scalers_path, 'r') as f:
                scalers_data = json.load(f)
                
            eval_data = None
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
        except json.JSONDecodeError as e:
            raise CommandError(f"Failed to parse JSON metadata: {e}")

        # 3. Resolve the Universe
        try:
            universe = Universe.objects.get(name=universe_slug)
        except Universe.DoesNotExist:
            self.stdout.write(self.style.WARNING(f"Universe '{universe_slug}' not found. Model will be saved without a universe link."))
            universe = None

        # 4. Extract architecture parameters
        # Defaulting to standard model if type isn't specified in legacy JSON
        model_type = arch_data.get('model_type', TrainedModel.STANDARD)
        
        try:
            seq_len = int(arch_data.get('seq_len', 60))
            d_model = int(arch_data.get('d_model', 64))
            n_heads = int(arch_data.get('n_heads', 4))
            n_layers = int(arch_data.get('n_layers', 2))
            n_features = int(arch_data.get('n_features', 49))
        except ValueError as e:
            raise CommandError(f"Invalid architecture parameter in JSON: {e}")

        # 5. Create the database record
        try:
            # Check if model already exists to update it, otherwise create new
            trained_model, created = TrainedModel.objects.get_or_create(
                name=model_name,
                defaults={
                    'model_type': model_type,
                    'seq_len': seq_len,
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'n_layers': n_layers,
                    'n_features': n_features,
                    'arch_json': arch_data,
                    'scalers_json': scalers_data,
                    'eval_json': eval_data,
                    'trained_on': universe
                }
            )

            if not created:
                self.stdout.write(self.style.WARNING(f"Model '{model_name}' already exists. Updating metadata..."))
                trained_model.model_type = model_type
                trained_model.seq_len = seq_len
                trained_model.d_model = d_model
                trained_model.n_heads = n_heads
                trained_model.n_layers = n_layers
                trained_model.n_features = n_features
                trained_model.arch_json = arch_data
                trained_model.scalers_json = scalers_data
                trained_model.eval_json = eval_data
                trained_model.trained_on = universe

            # Attach the actual .h5 weights file
            with open(weights_path, 'rb') as f:
                trained_model.weights_file.save(f"{model_name}.weights.h5", File(f), save=True)

            trained_model.save()
            
            action = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(f"Successfully {action.lower()} TrainedModel record for '{model_name}'"))
            self.stdout.write(self.style.SUCCESS(f"Weights file copied to: {trained_model.weights_file.name}"))

        except Exception as e:
            raise CommandError(f"Database error during model import: {e}")