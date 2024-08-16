In your project directory, create a file named .env and add your API key

```bash
WANDB_API_KEY=your_api_key
```

Then,


run the code with the following command:
```bash
python validator.py --model sklearn
```

In the future you will be able to change models by changing the argument of the --model flag. For now, only sklearn is supported due to size imports.