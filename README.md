# rossmann-sales

Test bench for Stan w/ ADVI and Pathfinder for scalable Bayesian inference. Example is taken from https://florianwilhelm.info/2020/10/bayesian_hierarchical_modelling_at_scale/. 


ToDo's
- [ ] Structured priors for weekdays
- [ ] 


To obtain the datasets, run the kaggle cli tool:

```
# check kaggle.json in ~/.kaggle/kaggle.json
kaggle datasets download pratyushakar/rossmann-store-sales -p data/raw/ --unzip
```