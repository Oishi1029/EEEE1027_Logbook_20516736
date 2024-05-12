import time

def my_function():
  print("Hello! Hope you're doing well")

def measure_time(func):
  start_time = time.time()
  func()
  end_time = time.time()
  print(f"Time taken by {func.__name__} function is {end_time - start_time} seconds")

measure_time(my_function)
