import logging
import sys

# Info Logger
logging.basicConfig(
    level=logging.INFO,
    # filename='program_logs/neural_network.log', 
    handlers=[
        logging.FileHandler('program_logs/neural_network.log'),
        logging.StreamHandler(sys.stdout)
    ],
    # filemode='w',
    format="%(asctime)s - %(levelname)s - %(message)s"
)