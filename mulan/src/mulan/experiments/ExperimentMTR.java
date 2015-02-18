package mulan.experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AverageRelativeRMSE;
import mulan.evaluation.measure.Measure;
import mulan.regressor.clus.ClusRandomForest;
import mulan.regressor.transformation.EnsembleOfRegressorChains;
import mulan.regressor.transformation.MultiTargetStacking;
import mulan.regressor.transformation.RegressorChain;
import mulan.regressor.transformation.SingleTargetRegressor;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.REPTree;
import weka.core.Utils;

/**
 * <p>
 * Class replicating the experiment in
 * <em>E. Spyromitros-Xioufis, G. Tsoumakas, W. Groves, I. Vlahavas. 2014. Multi-label Classification Methods for
 * Multi-target Regression. <a href="http://arxiv.org/abs/1211.6581">arXiv e-prints</a></em>.
 * </p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 *
 */
public class ExperimentMTR {

    /** the number of models in ensemble methods (ERC) **/
    public static final int numEnsembleModels = 10;
    /** whether the base learner should output debug messages **/
    public static final boolean baseDebug = false;
    /** whether the multi-target methods should output debug messages **/
    public static final boolean mtDebug = true;
    /** the number of cross-validation folds to use for evaluation **/
    public static final int numFolds = 10;
    /** the type of sampling in ERC **/
    public static final EnsembleOfRegressorChains.SamplingMethod sampling = EnsembleOfRegressorChains.SamplingMethod.None;
    /** number of execution slots to use by Weka's algorithms which support this option **/
    private static int numSlots;
    /** number of targets **/
    private static int numTargets;
    /** the multi-target datasets. Train and test will be null when cv is performed and vise versa **/
    private static MultiLabelInstances full;
    private static MultiLabelInstances train;
    private static MultiLabelInstances test;

    /**
     * @param args <ul>
     *            <li><b>-path:</b> full path to the dataset folder</li>
     *            <li><b>-filestem:</b> the dataset's filestem (name)</li>
     *            <li><b>-targets:</b> the number of targets in the dataset</li>
     *            <li><b>-eval:</b> the type of evaluation to perform ('cv' for cross-validation / 'train' for
     *            train/test split)</li>
     *            <li><b>-mt:</b> comma separated list of the multi-target regression methods to evaluate:<br>
     *            (ST,MTS,MTSC,ERC,ERCC,MORF)</li>
     *            <li><b>-base:</b>the base regressor to use (reptree-bag)</li>
     *            <li><b>-slots:</b> number of execution slots to be used by Weka's algorithms which support
     *            this option</li>
     *            </ul>
     * @throws Exception exceptions not caught
     */
    public static void main(String[] args) throws Exception {
        // parsing options related to dataset and evaluation type
        String path = Utils.getOption("path", args);
        String fileStem = Utils.getOption("filestem", args);
        numTargets = Integer.parseInt(Utils.getOption("targets", args));
        String evalType = Utils.getOption("eval", args);

        // parsing options related to multi-target methods being evaluated
        String mt = Utils.getOption("mt", args);
        String[] mtMethods = mt.split(",");
        String base = Utils.getOption("base", args);

        try {
            numSlots = Integer.parseInt(Utils.getOption("slots", args));
        } catch (Exception e) {
            System.out.println("Number of execution slots not specified, using 1.");
            numSlots = 1;
        }

        // loading the datasets
        if (evalType.startsWith("cv")) {
            full = new MultiLabelInstances(path + fileStem + ".arff", numTargets);
        } else {
            train = new MultiLabelInstances(path + fileStem + "-train.arff", numTargets);
            test = new MultiLabelInstances(path + fileStem + "-test.arff", numTargets);
            full = train; // just for initializing the measures
        }

        List<Measure> measures = new ArrayList<Measure>();
        measures.add(new AverageRelativeRMSE(numTargets, full, full));

        MultiLabelLearnerBase mtMethodPtr = null;

        String resultsFileName = "results_" + fileStem + evalType + "_" + mt + "_"
                + base.substring(0, 10) + "..." + base.substring(base.length() - 10, base.length())
                + ".txt";
        BufferedWriter outResults = new BufferedWriter(new FileWriter(resultsFileName));

        // header
        outResults
                .write("dataset\teval_type\tmt_method\tbase_learner\ttarget_index\ttarget_name\t");
        // print the measures name
        for (Measure m : measures) {
            outResults.write("'" + m.getName() + "'\t");
        }
        outResults.write("real_time\tcpu_time\n");
        outResults.flush();

        for (int j = 0; j < mtMethods.length; j++) { // for each mt method
            String mtMethodChoice = mtMethods[j];
            String baseLearnerChoice;
            Classifier baseLearner = null;
            if (mtMethodChoice.equals("MORF")) {
                baseLearnerChoice = "no";
            } else {
                baseLearnerChoice = base;
                baseLearner = selectBaseLearner(baseLearnerChoice);
            }
            if (mtMethodChoice.equals("ST")) {
                SingleTargetRegressor str = new SingleTargetRegressor(baseLearner);
                mtMethodPtr = str;
            } else if (mtMethodChoice.equals("MTS")) {
                MultiTargetStacking MTS = new MultiTargetStacking(baseLearner, baseLearner);
                MTS.setIncludeAttrs(true);
                MTS.setMeta(MultiTargetStacking.metaType.TRAIN);
                mtMethodPtr = MTS;
            } else if (mtMethodChoice.equals("MTSC")) {
                MultiTargetStacking MTSC = new MultiTargetStacking(baseLearner, baseLearner);
                MTSC.setIncludeAttrs(true);
                MTSC.setMeta(MultiTargetStacking.metaType.CV);
                MTSC.setNumFolds(10);
                mtMethodPtr = MTSC;
            } else if (mtMethodChoice.equals("ERC")) {
                EnsembleOfRegressorChains ERC = new EnsembleOfRegressorChains(baseLearner,
                        numEnsembleModels, sampling);
                ERC.setMeta(RegressorChain.metaType.TRUE);
                mtMethodPtr = ERC;
            } else if (mtMethodChoice.equals("ERCC")) {
                EnsembleOfRegressorChains ERCC = new EnsembleOfRegressorChains(baseLearner,
                        numEnsembleModels, sampling);
                ERCC.setMeta(RegressorChain.metaType.CV);
                ERCC.setNumFolds(10);
                mtMethodPtr = ERCC;
            } else if (mtMethodChoice.startsWith("MORF")) {
                ClusRandomForest MORF = new ClusRandomForest("clusWorkingDir/", fileStem, 100);
                mtMethodPtr = MORF;
            } else {
                throw new Exception(mtMethodChoice + " mt method is not supported!");
            }

            mtMethodPtr.setDebug(mtDebug);

            if (evalType.equals("train")) { // train-test evaluation
                long startTraining = System.currentTimeMillis();
                long startTrainingNano = getCpuTime();
                mtMethodPtr.build(train);
                long endTraining = System.currentTimeMillis();
                long endTrainingCPU = getCpuTime();

                Evaluator eval = new Evaluator();
                long startEval = System.currentTimeMillis();
                long startEvalNano = getCpuTime();
                Evaluation results = eval.evaluate(mtMethodPtr, test, train);
                long endEval = System.currentTimeMillis();
                long endEvalCPU = getCpuTime();

                AverageRelativeRMSE arrmse = (AverageRelativeRMSE) results.getMeasures().get(1);

                // print static information
                outResults.write(fileStem + "\t" + evalType + "\t" + mtMethodChoice + "\t"
                        + baseLearnerChoice + "\t0\tall\t");
                // print measure for all targets
                outResults.write(arrmse.getValue() + "\t");
                // print training/evaluation time
                outResults.write((endTraining - startTraining) + "\t"
                        + (endTrainingCPU - startTrainingNano) + "\t" + (endEval - startEval)
                        + "\t" + (endEvalCPU - startEvalNano) + "\n");

                // print measure per target
                for (int i = 0; i < numTargets; i++) {
                    outResults.write(fileStem + "\t" + evalType + "\t" + mtMethodChoice + "\t"
                            + baseLearnerChoice + "\t");
                    // print target index and name
                    outResults.write((i + 1) + "\t");
                    outResults.write(train.getDataSet().attribute(train.getLabelIndices()[i])
                            .name()
                            + "\t");
                    outResults.write(arrmse.getValue(i) + "\t");
                    // print training/evaluation time
                    outResults.write((endTraining - startTraining) + "\t"
                            + (endTrainingCPU - startTrainingNano) + "\t" + (endEval - startEval)
                            + "\t" + (endEvalCPU - startEvalNano) + "\n");
                }
                outResults.flush();

            } else if (evalType.equals("cv")) {
                Evaluator eval = new Evaluator();
                eval.setSeed(1);
                MultipleEvaluation results = null;

                long start = System.currentTimeMillis();
                long startCPU = getCpuTime();
                results = eval.crossValidate(mtMethodPtr, full, numFolds);
                long end = System.currentTimeMillis();
                long endCPU = getCpuTime();

                ArrayList<Evaluation> evals = results.getEvaluations();
                double[][] totalSEs = new double[numTargets][numFolds]; // a_i
                double[][] trainMeanTotalSEs = new double[numTargets][numFolds]; // b_i_us
                int[][] nonMissingInstances = new int[numTargets][numFolds];

                for (int t = 0; t < evals.size(); t++) { // for each fold!
                    AverageRelativeRMSE arrmse = ((AverageRelativeRMSE) evals.get(t).getMeasures()
                            .get(1));
                    for (int r = 0; r < numTargets; r++) {
                        totalSEs[r][t] = arrmse.getTotalSE(r);
                        trainMeanTotalSEs[r][t] = arrmse.getTrainMeanTotalSE(r);
                        // either measure can be used for getting the num non-missing
                        nonMissingInstances[r][t] = arrmse.getNumNonMissing(r);
                    }
                }

                // calculating rrmse
                double[] rrmse_us = new double[numTargets];
                for (int r = 0; r < numTargets; r++) {
                    for (int t = 0; t < numFolds; t++) {
                        rrmse_us[r] += Math.sqrt(totalSEs[r][t])
                                / Math.sqrt(trainMeanTotalSEs[r][t]);
                    }
                    rrmse_us[r] /= numFolds;
                }

                // print static information
                outResults.write(fileStem + "\t" + evalType + "\t" + mtMethodChoice + "\t"
                        + baseLearnerChoice + "\t0\tall\t");
                for (Measure m : measures) {
                    outResults.write(results.getMean(m.getName()) + "\t");
                }
                outResults.write((end - start) + "\t" + (endCPU - startCPU) + "\n");

                for (int m = 0; m < numTargets; m++) {
                    String targetName = full.getDataSet().attribute(full.getLabelIndices()[m])
                            .name();
                    outResults.write(fileStem + "\t" + evalType + "\t" + mtMethodChoice + "\t"
                            + baseLearnerChoice + "\t" + (m + 1) + "\t" + targetName + "\t");
                    for (Measure me : measures) {
                        outResults.write(results.getMean(me.getName(), m) + "\t");
                    }
                    outResults.write((end - start) + "\t" + (endCPU - startCPU) + "\n");
                }
                outResults.flush();
            } else {
                throw new Exception("Wrong evaluation type given!");
            }
        }
        outResults.close();

    }

    public static Classifier selectBaseLearner(String stLearnerName) throws Exception {
        Classifier stLearner = null;
        if (stLearnerName.equals("zeror")) {
            // Weka's ZeroR (mean predictor)
            ZeroR zeror = new ZeroR();
            stLearner = zeror;
        } else if (stLearnerName.equals("reptree-bag")) {
            // Bagging of 100 Weka's REPTrees (default)
            REPTree reptree = new REPTree();
            int numBags = 100;
            Bagging bagging = new Bagging();
            bagging.setNumIterations(numBags);
            bagging.setNumExecutionSlots(numSlots);
            bagging.setClassifier(reptree);
            stLearner = bagging;
        } else {
            throw new Exception(stLearnerName + " base learner is not supported!");
        }
        return stLearner;
    }

    /** 
     * Get CPU time in milliseconds.
     * @return the CPU time in ms
     */
    public static long getCpuTime() {
        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        return bean.isCurrentThreadCpuTimeSupported() ? (long) ((double) bean
                .getCurrentThreadCpuTime() / 1000000.0) : 0L;
    }

}
