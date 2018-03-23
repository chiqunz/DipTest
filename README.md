# Python Implementation of Hartigan's Dip Test

A python script to check if a sample is unimodal or multimodal. The algorithm is based on Hartigan's paper [The Dip Test of Unimodality](https://projecteuclid.org/euclid.aos/1176346577)

## Usage

Download the dip_test.py from codes, and call `$ dip()`. Parameter `num_bins` control the resolution and `p` control the confidence interval.

Return: the dip value and a boolean variable indicating whether the sample is unimodal or not