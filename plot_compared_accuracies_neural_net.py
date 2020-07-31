from plotting import plot_compared_accuracies

# the values below were simulated by Younes using feedforward neural networks in Keras
run0_training = 1
run0_train_run0_test = 0.94
run0_train_run1_test = 0.14
run0_train_mix_test = 0.55


run1_training= 0.98
run1_train_run0_test = 0.14
run1_train_run1_test = 0.85
run1_train_mix_test = 0.54

mix_training = 0.49
mix_train_run0_test = 0.44
mix_train_run1_test = 0.55
mix_train_mix_test = 0.46

plot_compared_accuracies(run0_train_run0_test, run0_train_run1_test, run0_train_mix_test, run1_train_run0_test, run1_train_run1_test,
                             run1_train_mix_test, mix_train_run0_test, mix_train_run1_test, mix_train_mix_test)
