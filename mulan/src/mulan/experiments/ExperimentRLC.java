package mulan.experiments;

import java.util.Date;
import java.util.Random;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.regressor.clus.ClusRandomForest;
import mulan.regressor.transformation.RandomLinearCombinations;
import mulan.regressor.transformation.SingleTargetRegressor;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * <p>
 * Class replicating the experiment in
 * <em>Grigorios Tsoumakas, Eleftherios Spyromitros-Xioufis, Aikaterini Vrekou, Ioannis Vlahavas. 2014. Multi-Target Regression via Random Linear Target
 * Combinations. <a href="http://arxiv.org/abs/1404.5065">arXiv e-prints</a></em>
 * </p>
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 * 
 */
public class ExperimentRLC {

    /** whether to output debug messages */
    public static boolean debug = false;
    /** number of folds when doing cross-validation */
    public static int numFolds = 10;
    /** a string representation of the base regressor */
    public static String baseRegressorChoice = "additive";

    /**
     * @param args <br>
     *            -path "path to the dataset folder" (required)<br>
     *            -filestem "dataset file name" (required)<br>
     *            -targets "number of targets in the dataset" (required)<br>
     *            -eval "evaluation type 'cv' and 'holdout' are supported (default: cv)"<br>
     *            -models "maximum number of models in the RLC method" (default: number of targets)<br>
     *            -seed "a seed number used for random number generation" (default: 1)<br>
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static void main(String[] args) throws Exception {
        String path = Utils.getOption("path", args);
        String fileStem = Utils.getOption("filestem", args);
        int numTargets = Integer.parseInt(Utils.getOption("targets", args));

        String evalType;
        try {
            evalType = Utils.getOption("eval", args);
            if (!evalType.equals("holdout") && !evalType.equals("cv")) {
                throw new Exception("Unknown evaluation type! 'cv' and 'holdout' are supported!");
            }
        } catch (Exception e) {
            System.out.println(e);
            evalType = "cv";
        }

        int numCombinations;
        try {
            numCombinations = Integer.parseInt(Utils.getOption("models", args));
            if (numCombinations < numTargets) {
                throw new IllegalArgumentException(
                        "Number of models should be at least as many as the number of target variables.");
            }
        } catch (Exception e) {
            System.out.println(e);
            numCombinations = numTargets;
        }

        long seed;
        try {
            seed = Long.parseLong(Utils.getOption("seed", args));
        } catch (Exception e) {
            System.out.println(e);
            seed = 1;
        }

        Classifier baseRegressorPtr = null;
        if (baseRegressorChoice.equals("additive")) {
            // Weka's Additive Regression with a small shrinkage rate (0.1) and a large number of
            // iterations (100) as suggested in "The Elements of Statistical Learning". REPTrees (unpruned)
            // with max tree depth = 2 (i.e. 4 terminal nodes) are used as base learners.
            AdditiveRegression ad = new AdditiveRegression();
            REPTree reptree = new REPTree();
            reptree.setNoPruning(true);
            reptree.setMaxDepth(2);
            ad.setClassifier(reptree);
            ad.setShrinkage(0.1);
            ad.setNumIterations(100);
            baseRegressorPtr = ad;
        } else {
            throw new Exception("Base regressor " + baseRegressorChoice + " is not supported!");
        }

        // the full dataset is loaded in all evaluation types
        MultiLabelInstances full = new MultiLabelInstances(path + fileStem + ".arff", numTargets);
        MultiLabelInstances mlTrain = null;
        MultiLabelInstances mlTest = null;
        if (evalType.equals("holdout")) { // train-test datasets are loaded only in holdout evaluation
            mlTrain = new MultiLabelInstances(path + fileStem + "-train.arff", numTargets);
            mlTest = new MultiLabelInstances(path + fileStem + "-test.arff", numTargets);
            numFolds = 1; // holdout evaluation is treated as cv with one fold (tricky)!
        }

        Evaluator eval = new Evaluator();
        Evaluation[][][] evaluationRLC = new Evaluation[numTargets - 1][numCombinations
                - numTargets + 1][numFolds];
        Evaluation[] evaluationST = new Evaluation[numFolds];
        Evaluation[] evaluationCLUS = new Evaluation[numFolds];

        // Normalize data. Target normalization is a requirement for meaningful linear combinations. Input
        // normalization should lead to faster convergence for some algorithms.
        Normalize normalize = new Normalize();
        normalize.setInputFormat(full.getDataSet());
        Instances workingSet = Filter.useFilter(full.getDataSet(), normalize);
        workingSet.randomize(new Random(seed));

        long STruntime = 0, CLUSruntime = 0, RLCruntime = 0;
        // num non-zero does not affect training time, i.e one measurement would suffice
        long RLCtrainTimes[] = new long[numTargets - 1];
        long RLCtestTimes[][] = new long[numTargets - 1][];

        System.err.println("" + new Date() + ": " + "RLC evaluation started");
        for (int i = 0; i < numFolds; i++) {
            // System.out.println("fold " + (i + 1) + "/" + numFolds);
            Instances train;
            Instances test;
            if (evalType.equals("cv")) {
                train = workingSet.trainCV(numFolds, i);
                test = workingSet.testCV(numFolds, i);
            } else { // holdout
                train = Filter.useFilter(mlTrain.getDataSet(), normalize);
                test = Filter.useFilter(mlTest.getDataSet(), normalize);
            }
            mlTrain = new MultiLabelInstances(train, full.getLabelsMetaData());
            mlTest = new MultiLabelInstances(test, full.getLabelsMetaData());
            // evaluate with different number of non-zero

            long start;
            long RLCstart = System.currentTimeMillis();
            for (int nonzero = 2; nonzero <= numTargets; nonzero++) {
                RandomLinearCombinations rlc = new RandomLinearCombinations(numCombinations, seed,
                        baseRegressorPtr, nonzero);
                rlc.setDebug(debug);
                // System.err.println("" + new Date() + ": " + "Building RLC with nonzero=" + nonzero
                // + " and numcombinations=" + numCombinations);
                start = System.currentTimeMillis();
                rlc.build(mlTrain);
                RLCtrainTimes[nonzero - 2] += System.currentTimeMillis() - start;
                // evaluate with different number of combinations
                RLCtestTimes[nonzero - 2] = new long[numCombinations - numTargets + 1];
                for (int j = numTargets; j < numCombinations; j++) {
                    // System.err.println("" + new Date() + ": " + "Evaluating RLC with nonzero="
                    // + nonzero + " and numcombinations=" + j);
                    rlc.setNumModels(j + 1);
                    start = System.currentTimeMillis();
                    evaluationRLC[nonzero - 2][j - numTargets][i] = eval.evaluate(rlc, mlTest,
                            mlTrain);
                    RLCtestTimes[nonzero - 2][j - numTargets] += System.currentTimeMillis() - start;
                }
            }
            RLCruntime += System.currentTimeMillis() - RLCstart;
        }
        for (int j = numTargets; j < numCombinations; j++) {
            for (int nonzero = 2; nonzero <= numTargets; nonzero++) {
                MultipleEvaluation me = new MultipleEvaluation(evaluationRLC[nonzero - 2][j
                        - numTargets], full);
                me.calculateStatistics();
                System.out.println(fileStem + ";" + "RLC" + ";" + baseRegressorChoice + ";"
                        + nonzero + ";" + (j + 1) + ";" + me.getMean("Average Relative RMSE") + ";"
                        + RLCruntime + ";" + RLCtrainTimes[nonzero - 2] + ";"
                        + RLCtestTimes[nonzero - 2][j - numTargets]);
            }
        }
        System.err.println("" + new Date() + ": " + "RLC evaluation completed");

        System.err.println("" + new Date() + ": " + "ST evaluation started");
        for (int i = 0; i < numFolds; i++) {
            // System.out.println("fold " + (i + 1) + "/" + numFolds);
            Instances train;
            Instances test;
            if (evalType.equals("cv")) {
                train = workingSet.trainCV(numFolds, i);
                test = workingSet.testCV(numFolds, i);
            } else { // holdout
                train = Filter.useFilter(mlTrain.getDataSet(), normalize);
                test = Filter.useFilter(mlTest.getDataSet(), normalize);
            }
            mlTrain = new MultiLabelInstances(train, full.getLabelsMetaData());
            mlTest = new MultiLabelInstances(test, full.getLabelsMetaData());
            SingleTargetRegressor st = new SingleTargetRegressor(baseRegressorPtr);
            long start = System.currentTimeMillis();
            st.build(mlTrain);
            evaluationST[i] = eval.evaluate(st, mlTest, mlTrain);
            STruntime += System.currentTimeMillis() - start;
        }
        if (evalType.equals("cv")) {
            MultipleEvaluation me = new MultipleEvaluation(evaluationST, full);
            me.calculateStatistics();
            System.out.println(fileStem + ";" + "ST" + ";" + baseRegressorChoice + ";" + "-" + ";"
                    + "-" + ";" + me.getMean("Average Relative RMSE") + ";" + STruntime);
        } else {
            System.out
                    .println(fileStem + ";" + "ST" + ";" + baseRegressorChoice + ";" + "-" + ";"
                            + "-" + ";" + evaluationST[0].getMeasures().get(1).getValue() + ";"
                            + STruntime);
        }
        System.err.println("" + new Date() + ": " + "ST evaluation completed");

        System.err.println("" + new Date() + ": " + "CLUS evaluation started");
        for (int i = 0; i < numFolds; i++) {
            // System.out.println("fold " + (i + 1) + "/" + numFolds);
            Instances train;
            Instances test;
            if (evalType.equals("cv")) {
                train = workingSet.trainCV(numFolds, i);
                test = workingSet.testCV(numFolds, i);
            } else { // holdout
                train = Filter.useFilter(mlTrain.getDataSet(), normalize);
                test = Filter.useFilter(mlTest.getDataSet(), normalize);
            }
            mlTrain = new MultiLabelInstances(train, full.getLabelsMetaData());
            mlTest = new MultiLabelInstances(test, full.getLabelsMetaData());
            ClusRandomForest clus = new ClusRandomForest("clusWorkingDir/", fileStem, 100);
            long start = System.currentTimeMillis();
            clus.build(mlTrain);
            evaluationCLUS[i] = eval.evaluate(clus, mlTest, mlTrain);
            CLUSruntime += System.currentTimeMillis() - start;
        }
        if (evalType.equals("cv")) {
            MultipleEvaluation me = new MultipleEvaluation(evaluationCLUS, full);
            me.calculateStatistics();
            System.out.println(fileStem + ";" + "CLUS-rforest" + ";" + "-" + ";" + "-" + ";" + "-"
                    + ";" + me.getMean("Average Relative RMSE") + ";" + CLUSruntime);
        } else {
            System.out.println(fileStem + ";" + "CLUS-rforest" + ";" + "-" + ";" + "-" + ";" + "-"
                    + ";" + evaluationCLUS[0].getMeasures().get(1).getValue() + ";" + CLUSruntime);
        }
        System.err.println("" + new Date() + ": " + "CLUS evaluation completed");

    }
}
