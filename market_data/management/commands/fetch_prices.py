from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = "Fetches historical market data and populates the database."

    def add_arguments(self, parser):
        parser.add_argument('--universe', type=str, required=True, help='Name of the universe')
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')

    def handle(self, *args, **options):
        universe_name = options['universe']
        start_date = options['start']
        
        self.stdout.write(f"Fetching data for universe '{universe_name}' starting from {start_date}...")
        
        # Your core price-fetching logic goes here (or call your service function)
        
        self.stdout.write(self.style.SUCCESS("Successfully fetched market data!"))