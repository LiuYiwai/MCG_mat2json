# MCG_mat2json
Make Multiscale Combinatorial Grouping(MCG) data change from mat to json

## How to use ?

Please edit the code to modify the parameters, such as base_ dir, name_ path, json_ path, data_path, thread_num, global_limit.



The file directory is as follows:

```
base_dir
    ├── json_path
    ├── data_path
    ├── name_path
```

name_path defaults to trainval.txt get from VOC2012/ImageSets/Main/trainval.txt 

global_limit means the maximum number of proposals per image. 

