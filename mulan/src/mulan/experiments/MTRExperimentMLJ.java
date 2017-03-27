package mulan.experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.regression.macro.MacroRelRMSE;
import mulan.regressor.clus.ClusRandomForest;
import mulan.regressor.clus.ClusWrapperRegression;
import mulan.regressor.malsar.Dirty;
import mulan.regressor.malsar.TraceNormRegularization;
import mulan.regressor.transformation.EnsembleOfRegressorChains;
import mulan.regressor.transformation.MultiTargetStacking;
import mulan.regressor.transformation.RandomLinearCombinationsNormalize;
import mulan.regressor.transformation.RegressorChain;
import mulan.regressor.transformation.SingleTargetRegressor;
import weka.classifiers.Classifier;
import weka.classifiers.StohasticGradientBoosting;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.GridSearch9734Mod;
import weka.classifiers.trees.REPTree;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.filters.AllFilter;

/**
 * This class replicates the results of the paper:<br>
 * Spyromitros-Xioufis, E., Tsoumakas, G., Groves, W., Vlahavas, I. Mach Learn (2016)
 * 104: 55. doi:10.1007/s10994-016-5546-z.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 */
public class MTRExperimentMLJ {

	/**
	 * @param args
	 *            <ul>
	 *            <li><b>-path:</b> full path to the dataset folder</li>
	 *            <li><b>-filestem:</b> the dataset's filestem (name)</li>
	 *            <li><b>-targets:</b> the number of targets in the dataset</li>
	 *            <li><b>-eval:</b> the type of evaluation to perform ('cv-k' for k-fold cross-validation / 'holdout'
	 *            for holdout evaluation)</li>
	 *            <li><b>-mt:</b> comma separated list of the multi-target regression methods to evaluate: ST,
	 *            SST/ERC_{true/train/cv}, MORF, TNR, Dirty, RLC</li>
	 *            <li><b>-base:</b> comma separated list of the base regressors to use (tree/bag/svr/ridge/sgb)</li>
	 *            <li><b>-malsar:</b> full path to malsar's Matlab implementation (required only if a malsar method is
	 *            evaluated)</li>
	 *            <li><b>-slots:</b> number of execution slots to be used by Weka's/Malsar's algorithms that support
	 *            this option (optional, default=1)</li>
	 *            </ul>
	 *            Example set of parameters: -path "data/" -filestem
	 *            "solar-flare_1" -targets 3 -eval "cv-10" -mt "ST,ERC_{true}"
	 *            -base "bag" -slots 2
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// parsing options related to dataset and evaluation type
		String path = Utils.getOption("path", args);
		fileStem = Utils.getOption("filestem", args);
		int numTargets = Integer.parseInt(Utils.getOption("targets", args));
		String evalType = Utils.getOption("eval", args);
		// parsing options related to multi-target methods and base regressors
		String mt = Utils.getOption("mt", args);
		String[] mtMethods = mt.split(",");
		String base = Utils.getOption("base", args);
		String[] baseLearners = base.split(",");
		try {
			numSlots = Integer.parseInt(Utils.getOption("slots", args));
		} catch (Exception e) {
			System.out.println("Number of execution slots not specified, using 1.");
			numSlots = 1;
		}

		// put all CLUSMethods in a HashSet
		HashSet<String> clusMethods = new HashSet<String>();
		for (String clusMethod : CLUSMethodsArray) {
			clusMethods.add(clusMethod);
		}
		// put all malsarMethods in a HashSet
		HashSet<String> malsarMethods = new HashSet<String>();
		for (String malsarMethod : MALSARMethodsArray) {
			malsarMethods.add(malsarMethod);
		}

		if (containsMethod(mtMethods, malsarMethods)) {// the path to malsar's Matlab implementation is need
			malsarPath = Utils.getOption("malsar", args);
			if (malsarPath.length() > 0) { // malsar's path was set
				MatlabProxyFactoryOptions options = new MatlabProxyFactoryOptions.Builder().setHidden(true).build();
				MatlabProxyFactory factory = new MatlabProxyFactory(options);
				matlabProxy = factory.getProxy();
			} else {
				throw new Exception("A malsar method was selected but the '-malsar' option was not set!");
			}
		}

		// loading the datasets
		// The multi-target datasets. Train and test will be null when cv is performed and vise versa.
		MultiLabelInstances full = null;
		MultiLabelInstances train = null;
		MultiLabelInstances test = null;
		int numFolds = 0;
		if (evalType.startsWith("cv")) {
			numFolds = Integer.parseInt(evalType.split("-")[1]);
			full = new MultiLabelInstances(path + fileStem + ".arff", numTargets);
		} else if (evalType.equals("holdout")) {
			train = new MultiLabelInstances(path + fileStem + "-train.arff", numTargets);
			test = new MultiLabelInstances(path + fileStem + "-test.arff", numTargets);
			full = train; // just for initializing the measures
		} else {
			throw new Exception("Uknown evaluation type!");
		}

		List<Measure> measures = new ArrayList<Measure>();
		measures.add(new MacroRelRMSE(full, full));

		String baseName = base;
		if (base.length() > 15) { // if the base string is too long, create a short version
			baseName = base.substring(0, 6) + "..." + base.substring(base.length() - 6, base.length());
		}

		String mtName = mt;
		if (mt.length() > 15) { // if the mt string is too long, create a short version
			mtName = mt.substring(0, 6) + "..." + mt.substring(mt.length() - 6, mt.length());
		}

		String resultsFileName = "res_" + fileStem + "_" + evalType + "_" + mtName + "_" + baseName + "_" + numSlots
				+ ".txt";

		BufferedWriter outResults = new BufferedWriter(new FileWriter(resultsFileName));
		String colSep = "\t";
		// print the header
		outResults.write("dataset" + colSep + "eval_type" + colSep + "mt_method" + colSep + "base_learner" + colSep
				+ "target_index" + colSep + "target_name" + colSep + "RRMSE" + colSep + "RRMSE_Aho" + colSep);

		if (evalType.startsWith("cv")) {
			outResults.write("time" + colSep + "numslots" + "\n");
		} else {
			outResults.write("time_tr" + colSep + "time_te" + colSep + "numSlots\n");
		}
		outResults.flush();

		for (int j = 0; j < mtMethods.length; j++) { // for each mt method
			for (int k = 0; k < baseLearners.length; k++) { // for each base learner
				String mtMethodChoice = mtMethods[j];
				String baseLearnerChoice = baseLearners[k];
				Classifier baseLearner = null;
				if (clusMethods.contains(mtMethodChoice) || malsarMethods.contains(mtMethodChoice)) {
					baseLearnerChoice = "no"; // an adaptation method
				} else {
					baseLearner = selectBaseRegressor(baseLearnerChoice);
				}
				MultiLabelLearner mtMethodPtr = selectMTRMethod(mtMethodChoice, baseLearner);

				String staticInfo = fileStem + colSep + evalType + colSep + mtMethodChoice + colSep + baseLearnerChoice
						+ colSep;

				if (evalType.equals("train")) { // train-test evaluation

					long startTrainingReal = System.currentTimeMillis();
					mtMethodPtr.build(train);
					long endTrainingReal = System.currentTimeMillis();

					Evaluator eval = new Evaluator();
					long startEvalReal = System.currentTimeMillis();
					Evaluation results = eval.evaluate(mtMethodPtr, test, train);
					long endEvalReal = System.currentTimeMillis();

					String timeMeasurements = (endTrainingReal - startTrainingReal) + colSep
							+ (endEvalReal - startEvalReal);

					// create average result line
					String resLineAll = staticInfo + "0" + colSep + "all" + colSep;
					for (Measure m : measures) {
						resLineAll += m.getValue() + colSep;
					}
					// append train/test time measurements
					resLineAll += timeMeasurements;
					outResults.write(resLineAll + "\n");
					// print measures per target
					for (int i = 0; i < numTargets; i++) {
						String resLineTarget = staticInfo;
						// append target index and name
						String targetName = train.getDataSet().attribute(train.getLabelIndices()[i]).name();
						resLineTarget += (i + 1) + colSep + targetName + colSep;
						for (Measure m : measures) {
							double targetValue = -1; // default for non macro measures
							if (m instanceof MacroAverageMeasure) {
								targetValue = ((MacroAverageMeasure) m).getValue(i);
							}
							resLineTarget += targetValue + colSep;
						}
						// append train/test time measurements
						resLineTarget += timeMeasurements + colSep + numSlots;
						outResults.write(resLineTarget + "\n");
					}
					outResults.flush();
				} else if (evalType.startsWith("cv")) {
					Evaluator eval = new Evaluator();
					MultipleEvaluation results = null;

					long start = System.currentTimeMillis();
					results = eval.crossValidate(mtMethodPtr, full, numFolds);
					long end = System.currentTimeMillis();

					String timeMeasurements = (end - start) + colSep;

					ArrayList<Evaluation> evals = results.getEvaluations();
					double[][] totalSEs = new double[numTargets][numFolds];
					double[][] trainMeanTotalSEs = new double[numTargets][numFolds];
					double[][] fullMeanTotalSEs = new double[numTargets][numFolds];
					int[][] nonMissingInstances = new int[numTargets][numFolds];

					for (int t = 0; t < evals.size(); t++) { // for each fold
						MacroRelRMSE arrmse = ((MacroRelRMSE) evals.get(t).getMeasures().get(1));
						for (int r = 0; r < numTargets; r++) {
							totalSEs[r][t] = arrmse.getTotalSE(r);
							trainMeanTotalSEs[r][t] = arrmse.getTrainMeanTotalSE(r);
							fullMeanTotalSEs[r][t] = arrmse.getFullMeanTotalSE(r);
							nonMissingInstances[r][t] = arrmse.getNonMissing(r);
						}
					}

					// calculating measures, our way
					double[] rmse_us = new double[numTargets];
					double[] rrmse_us = new double[numTargets];
					for (int r = 0; r < numTargets; r++) {
						for (int t = 0; t < numFolds; t++) {
							rmse_us[r] += Math.sqrt(totalSEs[r][t] / nonMissingInstances[r][t]);
							rrmse_us[r] += Math.sqrt(totalSEs[r][t]) / Math.sqrt(trainMeanTotalSEs[r][t]);

						}
						rmse_us[r] /= numFolds;
						rrmse_us[r] /= numFolds;
					}

					// calculating measures, Aho's way
					double[] rmse_aho = new double[numTargets];
					double[] rrmse_aho = new double[numTargets];
					for (int r = 0; r < numTargets; r++) {
						rmse_aho[r] = Math.sqrt(Utils.sum(totalSEs[r]) / Utils.sum(nonMissingInstances[r]));
						rrmse_aho[r] = rmse_aho[r]
								/ Math.sqrt(Utils.sum(fullMeanTotalSEs[r]) / Utils.sum(nonMissingInstances[r]));
					}

					HashMap<String, double[]> measuresAho = new HashMap<String, double[]>();
					measuresAho.put(measures.get(0).getName(), rrmse_aho);

					// create result line for all targets
					String resLineAll = staticInfo + "0" + colSep + "all" + colSep;
					for (Measure m : measures) {
						resLineAll += results.getMean(m.getName()) + colSep;
						resLineAll += Utils.mean(measuresAho.get(m.getName())) + colSep;
					}

					resLineAll += timeMeasurements + colSep + numSlots;
					outResults.write(resLineAll + "\n");

					for (int m = 0; m < numTargets; m++) {
						String resLineTarget = staticInfo;
						String targetName = full.getDataSet().attribute(full.getLabelIndices()[m]).name();
						resLineTarget += (m + 1) + colSep + targetName + colSep;

						for (Measure me : measures) {
							double targetValue = -1; // default for non macro measures
							if (me instanceof MacroAverageMeasure) {
								targetValue = results.getMean(me.getName(), m);
							}
							resLineTarget += targetValue + colSep;
							// once again for Aho style calculation
							targetValue = measuresAho.get(me.getName())[m];
							resLineTarget += targetValue + colSep;

						}
						resLineTarget += (end - start) + colSep + numSlots;
						outResults.write(resLineTarget + "\n");
					}
					outResults.flush();

				} else {
					throw new Exception("Wrong evaluation type given!");
				}

				// no point in running CLUS/Malsar methods with different base classifiers
				if (clusMethods.contains(mtMethodChoice) || malsarMethods.contains(mtMethodChoice)) {
					break;
				}
			}
		}

		outResults.close();
		if (matlabProxy != null) {
			matlabProxy.exit();
		}

	}

	/** Methods supported from CLUS. */
	public static final String[] CLUSMethodsArray = { "MORF" };
	/** Methods supported from Malsar. */
	public static final String[] MALSARMethodsArray = { "TNR", "Dirty" };
	/** Number of execution slots for Weka algorithms that support multi-threading. **/
	private static int numSlots;
	/** This is needed for starting Matlab from Java. */
	public static MatlabProxy matlabProxy;
	/** This is the path to Malsar's sources. */
	public static String malsarPath;
	/** The dataset's filestem */
	public static String fileStem;

	private static boolean containsMethod(String[] methods, HashSet<String> referenceMethods) {
		for (String method : methods) {
			if (referenceMethods.contains(method)) {
				return true;
			}
		}
		return false;
	}

	public static MultiLabelLearner selectMTRMethod(String mtMethodChoice, Classifier baseLearner) throws Exception {
		MultiLabelLearner mtMethodPtr;
		if (mtMethodChoice.equals("ST")) {
			SingleTargetRegressor str = new SingleTargetRegressor(baseLearner);
			mtMethodPtr = str;
		} else if (mtMethodChoice.startsWith("SST")) {
			Classifier firstStage = baseLearner;
			Classifier secondStage = baseLearner; // the same learner is used for both stages
			MultiTargetStacking SST = new MultiTargetStacking(firstStage, secondStage);
			SST.setIncludeAttrs(true);

			String mtMethodVariant = mtMethodChoice.split("_")[1];

			if (mtMethodVariant.equals("{true}")) {
				SST.setMeta(MultiTargetStacking.metaType.TRUE);
			} else if (mtMethodVariant.equals("{train}")) {
				SST.setMeta(MultiTargetStacking.metaType.INSAMPLE);
			} else if (mtMethodVariant.equals("{cv}")) {
				SST.setMeta(MultiTargetStacking.metaType.CV);
				SST.setNumFolds(10);
			} else {
				throw new Exception(mtMethodChoice + " SST variant does not exist!");
			}
			mtMethodPtr = SST;
		} else if (mtMethodChoice.startsWith("ERC")) {
			EnsembleOfRegressorChains ERC = new EnsembleOfRegressorChains(baseLearner, 10,
					EnsembleOfRegressorChains.SamplingMethod.None);

			String mtMethodVariant = mtMethodChoice.split("_")[1];

			if (mtMethodVariant.equals("{true}")) {
				ERC.setMeta(RegressorChain.metaType.TRUE);
			} else if (mtMethodVariant.equals("{train}")) {
				ERC.setMeta(RegressorChain.metaType.INSAMPLE);
			} else if (mtMethodVariant.equals("{cv}")) {
				ERC.setMeta(RegressorChain.metaType.CV);
				ERC.setNumFolds(10);
			} else {
				throw new Exception(mtMethodChoice + " ERC variant does not exist!");
			}
			mtMethodPtr = ERC;
		} else if (mtMethodChoice.equals("MORF")) {
			ClusWrapperRegression clus = new ClusRandomForest("clusWorkingDir/", fileStem, 100);
			mtMethodPtr = clus;
		} else if (mtMethodChoice.equals("TNR")) {
			TraceNormRegularization trace = new TraceNormRegularization(malsarPath, matlabProxy, numSlots);
			trace.setNumCVFolds(5);
			mtMethodPtr = trace;
		} else if (mtMethodChoice.equals("Dirty")) {
			Dirty dirty = new Dirty(malsarPath, matlabProxy, numSlots);
			dirty.setNumCVFolds(5);
			mtMethodPtr = dirty;
		} else if (mtMethodChoice.equals("RLC")) {
			// parameters of the RLC method
			int RLCNonZeroPerRow = 2;
			int RLCNumCombinations = 100;
			int RLCSeed = 1;
			RandomLinearCombinationsNormalize RLC = new RandomLinearCombinationsNormalize(RLCNumCombinations, RLCSeed,
					baseLearner, RLCNonZeroPerRow);
			mtMethodPtr = RLC;
		} else {
			throw new Exception(mtMethodChoice + " MTR method does not exist!");
		}
		mtMethodPtr.setDebug(true);
		return mtMethodPtr;

	}

	public static Classifier selectBaseRegressor(String baseRegressorName) throws Exception {
		Classifier stLearner = null;
		if (baseRegressorName.equals("tree")) { // Weka's REPTree
			REPTree reptree = new REPTree();
			stLearner = reptree;
		} else if (baseRegressorName.equals("bag")) { // Bagging of 100 Weka's REPTrees
			REPTree reptree = new REPTree();
			int numBags = 100;
			Bagging bagging = new Bagging();
			bagging.setNumIterations(numBags);
			bagging.setNumExecutionSlots(numSlots);
			bagging.setClassifier(reptree);
			stLearner = bagging;
		} else if (baseRegressorName.equals("ridge")) {
			// A realization of ridge regression using a modified version Weka's GridSearch (which
			// allows setting an arbitrary number of folds for the initial grid) to tune the ridge
			// parameter of LinearRegression (minimizing RMSE). We test values in the range
			// 10^-4...10^2. Because GridSearch is designed for tuning 2 parameters, we also include
			// the boolean parameter eliminateColinearAttributes.
			LinearRegression lr = new LinearRegression();
			lr.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE,
					LinearRegression.TAGS_SELECTION));
			lr.setEliminateColinearAttributes(false);
			lr.setOutputAdditionalStats(false);
			lr.setMinimal(true);

			GridSearch9734Mod grid = initializeGridSearch();
			grid.setFilter(new AllFilter()); // this filter is equal to not using a filter
			grid.setClassifier(lr);

			grid.setYProperty("classifier.ridge");
			grid.setYMin(-4);
			grid.setYMax(2);
			grid.setYStep(1);
			grid.setYExpression("pow(BASE,I)");
			grid.setYBase(10);

			grid.setXProperty("classifier.eliminateColinearAttributes");
			grid.setXMin(0);
			grid.setXMax(1);
			grid.setXStep(1);
			grid.setXExpression("I");

			stLearner = grid;

		} else if (baseRegressorName.equals("svr")) {
			LibLINEAR liblinear = new LibLINEAR();
			liblinear.setSVMType(new SelectedTag(11, LibLINEAR.TAGS_SVMTYPE));

			GridSearch9734Mod grid = initializeGridSearch();
			grid.setFilter(new AllFilter()); // this filter is equal to not using a filter
			grid.setClassifier(liblinear);

			grid.setYProperty("classifier.cost");
			grid.setYMin(-4);
			grid.setYMax(2);
			grid.setYStep(1);
			grid.setYExpression("pow(BASE,I)");
			grid.setYBase(10);
			// below we use the default 1000 iterations, as well as a dummy value of 1 (to finish quickly)
			grid.setXProperty("classifier.maximumNumberOfIterations");
			grid.setXMin(1);
			grid.setXMax(1000);
			grid.setXStep(999);
			grid.setXExpression("I");

			stLearner = grid;
		} else if (baseRegressorName.equals("sgb")) {
			// Stochastic Gradient Boosting with a small shrinkage rate (0.1) and a large number of iterations (100).
			// REPTrees (unpruned) with max tree depth = 2 are used as base learners.
			StohasticGradientBoosting sgb = new StohasticGradientBoosting();
			REPTree reptree = new REPTree();
			reptree.setNoPruning(true);
			reptree.setMaxDepth(2);
			sgb.setClassifier(reptree);
			sgb.setShrinkage(0.1);
			sgb.setNumIterations(100);
			sgb.setPercentage(66);
			stLearner = sgb;
		} else {
			throw new Exception(baseRegressorName + " base learner is not supported!");
		}
		return stLearner;
	}

	/**
	 * Initializes GridSearch with options that are common among all tunable classifiers. This is a modified version of
	 * Weka's GridSearch that allows setting the number of cv folds and performs parameter search only once.
	 * 
	 * @return
	 */
	public static GridSearch9734Mod initializeGridSearch() {
		GridSearch9734Mod grid = new GridSearch9734Mod();
		// the metric to optimize
		grid.setEvaluation(new SelectedTag(GridSearch9734Mod.EVALUATION_RMSE, GridSearch9734Mod.TAGS_EVALUATION));
		grid.setGridIsExtendable(false);
		grid.setNumExecutionSlots(numSlots);
		grid.setSampleSizePercent(100);
		grid.setInitialNumFolds(5);
		grid.setStopAfterFirstGrid(true);
		grid.setTraversal(new SelectedTag(GridSearch9734Mod.TRAVERSAL_BY_ROW, GridSearch9734Mod.TAGS_TRAVERSAL));
		return grid;
	}

}
