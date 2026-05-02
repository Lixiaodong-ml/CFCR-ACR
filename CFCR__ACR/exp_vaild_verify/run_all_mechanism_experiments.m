clearvars;
close all;
clc;

addpath(fileparts(mfilename('fullpath')));
mech.utils.ensurePaths();

fprintf('Running all mechanism experiments under exp_vaild_verify...\n');

exp01_matched_synthetic;
exp02_boundary_easy_hard;
exp03_tail_sensitive;
exp04_geometry_diagnostics;
exp05_disagreement_conditioned_gain;
exp06_frozen_refined_descriptor_ablation;
exp07_scalar_vs_coordinatewise;
exp08_hellinger_vs_alternatives;
exp09_mismatched_negative_control;
exp10_approximate_mismatch_scan;
exp11_real_classifier_mechanism;

fprintf('All mechanism experiments finished.\n');
