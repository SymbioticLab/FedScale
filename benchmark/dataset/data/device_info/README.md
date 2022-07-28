# Device Trace File Formats

Currently, FedScale supports plugging in two types of device traces in the simulation mode. 

The first is a client device computation and communication capacity trace, where each entry is a key-value pair of client-id and a value in the following format.
```
{
  'computation': FP32, 
  'communication' FP32: 
}
```

The second is client availability trace, where key is the client-id and value is in the following format.
```
{
  'duration': INT, 
  'inactive': [INT], 
  'finish_time': INT, 
  'active': [INT], 
  'model': STRING
}
```

FedScale comes with one trace for each type (`client_device_capacity` and `client_behave_trace`) that can, of course, be replaced.
Both files are dictionaries of entries with the formats outlined above and have been serialized using `pickle`.
