import os
from datetime import datetime

def create_text_file():
    directory = "experiments/results/tests"
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"it_worked_{timestamp}.txt"
    
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, 'w') as file:
        file.write("it worked")
    
    print(f'The file "{file_name}" has been created successfully in "{directory}".')

create_text_file()
