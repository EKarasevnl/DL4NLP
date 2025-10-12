This marks the storage location of the Flores+ dataset.
Follow the steps here to access it: https://huggingface.co/datasets/openlanguagedata/flores_plus <br>
It will ask you to run:
```
import huggingface_hub
from datasets import load_dataset

huggingface_hub.login()
ds_full = load_dataset("openlanguagedata/flores_plus")
```
Then save it locally by 
```
running ds_full.save_to_disk("data/flores_plus")
```