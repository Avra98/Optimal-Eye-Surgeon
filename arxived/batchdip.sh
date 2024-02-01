python -u deep_hess_batch.py --max_steps=8000 --optim="SGD" --sigma=0.1 --device_id=0 &
python -u -deep_hess_batch.py -max_steps=8000 --optim="SGD" --sigma=0.05 --device_id=2 &
python -u deep_hess_batch.py --max_steps=8000 --optim="SAM" --sigma=0.1 --device_id=4 &
python -u deep_hess_batch.py --max_steps=8000 --optim="SAM" --sigma=0.05 --device_id=6




