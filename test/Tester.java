import java.io.*;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.LabelBasedEvaluation;
import mulan.classifier.LabelPowersetClassifier;
import mulan.classifier.FlattenTrueLabelsClassifier;
import mulan.classifier.RAKEL;

import weka.core.*;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.*;

public class Tester
{
	

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception
	{
		String path = "c:/work/datasets/multilabel/";
		/*
		String datastem = "mediamill1";
		double[] subsetSizes = {2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100};		
		int numLabels = 101;
		*/
		//*
                String directory = "tmc2007";
		String datastem = "tmc2007-500";
		int numLabels = 22;
                SMO baseClassifier  = new SMO();               
                /* 
                weka.classifiers.functions.LibSVM baseClassifier = new weka.classifiers.functions.LibSVM();
                Tag[] tg = new Tag[1];
                tg[0] = new Tag(1, "KERNELTYPE_LINEAR");
                SelectedTag st = new SelectedTag(1, tg);
                baseClassifier.setKernelType(st);
                //NaiveBayes baseClassifier = new NaiveBayes();
		//*/
		/*
                String directory = "rcv1v2/subset";
		String datastem = "rcv1-subset1";
                SMO baseClassifier  = new SMO();
                /*
                weka.classifiers.functions.LibSVM baseClassifier = new weka.classifiers.functions.LibSVM();
                Tag[] tg = new Tag[1];
                tg[0] = new Tag(1, "KERNELTYPE_LINEAR");
                SelectedTag st = new SelectedTag(1, tg);
                baseClassifier.setKernelType(st);
                */
                //NaiveBayes baseClassifier = new NaiveBayes();
		//int numLabels = 101;
		/*/		/*
		String directory = "yeast";
		String datastem = "yeast";
                SMO baseClassifier = new SMO();
		int numLabels = 14;
		//*/
		/*
		String datastem = "scene";
                String directory = "scene";
                SMO baseClassifier = new SMO();
		int numLabels = 6;
		//*/
		FileReader frTrain = new FileReader(path + directory + "/" + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + directory + "/" + datastem + "-test.arff");
		Instances testData = new Instances(frTest);
		//FileReader frAll = new FileReader(path + directory + "/" + directory + ".arff");
		//Instances allData = new Instances(frAll);

		//int avgModelsPerLabel = 5;	
		//SMO baseClassifier = new SMO();
		//IBk baseClassifier = new IBk();
		//baseClassifier.setKNN(9);
               // NaiveBayes baseClassifier = new NaiveBayes();
		//double[] subsetSizes = {2, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100};

		//Statistics stats = new Statistics(allData, numLabels);
                //stats.showStats();
		/*
  	    Evaluator eval;
		Evaluation results;
		System.out.println("RSC");
		for (int i=0; i<subsetSizes.length; i++) {
			System.out.println("Subset Size: " + subsetSizes[i]);
			RAKEL rsc = new RAKEL();
			rsc.setNumLabels(numLabels);
			rsc.setBaseClassifier(Classifier.makeCopy(baseClassifier));
			rsc.setSizeOfSubset((int) subsetSizes[i]);
			rsc.setNumModels((int) (avgModelsPerLabel*numLabels/subsetSizes[i]));
			System.out.println("Num Models : " + (int) (avgModelsPerLabel*numLabels/subsetSizes[i]));
			rsc.buildClassifier(trainData);
			for (double t=0.1; t<=0.9; t+=0.1) {
				rsc.setExtThreshold(t);
				eval = new Evaluator();
				results = eval.evaluate(rsc, testData);
				System.out.println("HammingLoss t=" + t + " : " + results.getExampleBased().hammingLoss());
				// what other measures.... + correct them appropriately
			}
		}
		//*/
		
                /*
		int numRep=100;
                int numModels=1; 
		int subsetSize=2;                     
                Statistics stats = new Statistics(trainData, numLabels);
                stats.calculateCoocurrence(trainData, numLabels);
                for (int k=0; k<numRep; k++) {
                    RAKEL rsc = new RAKEL(numLabels, numModels, subsetSize);
                    rsc.setBaseClassifier(Classifier.makeCopy(baseClassifier));
                    for (int i=0; i<numModels; i++) {
                            rsc.updateClassifier(trainData, i);
                            Evaluator evaluator = new Evaluator();
                            rsc.updatePredictions(testData, i);
                            rsc.subsetClassifiers[i] = null;
                            Evaluation[] results = evaluator.evaluateOverThreshold(rsc.getPredictions(), testData, 0.1, 0.1, 9);
                            //Evaluation[] results = evaluator.evaluateOverThreshold(rsc, testData, 0.1, 0.1, 9);
                            for (int t=0; t<results.length; t++) {
                                    results[t].getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
                                    System.out.println("model=" + i + ";t=0." + (t+1) + 
                                                       ";hl=" + results[t].getExampleBased().hammingLoss() +
                                                       ";pr=" + results[t].getLabelBased().precision() + 
                                                       ";re=" + results[t].getLabelBased().recall() +
                                                       ";f1=" + results[t].getLabelBased().fmeasure());
                            }
                            // what other measures.... + correct them appropriately
                    }	
                }	                                 
                //*/
                
                
		/* Specific Subset Size All Combinations Experiment
		System.out.println("RSC");
		int numModels=200; // 
		System.out.println("Num Models : " + numModels);
		int subsetSize=9;   // 
		System.out.println("Subset Size: " + subsetSize);
		RAKEL rsc = new RAKEL(numLabels, numModels, subsetSize);
		PrintWriter pw = new PrintWriter("rkl-" + datastem + "-k" + subsetSize + ".txt");                      
		rsc.setBaseClassifier(Classifier.makeCopy(baseClassifier));
		for (int i=0; i<numModels; i++) {
			rsc.updateClassifier(trainData, i);
			Evaluator evaluator = new Evaluator();
			rsc.updatePredictions(testData, i);
                        rsc.nullSubsetClassifier(i);
			Evaluation[] results = evaluator.evaluateOverThreshold(rsc.getPredictions(), testData, 0.1, 0.1, 9);
			for (int t=0; t<results.length; t++) {
				results[t].getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
				pw.println("model=" + i + ";t=0." + (t+1) + 
						   ";hl=" + results[t].getExampleBased().hammingLoss() +
						   ";pr=" + results[t].getLabelBased().precision() + 
						   ";re=" + results[t].getLabelBased().recall() +
                                                   ";f1=" + results[t].getLabelBased().fmeasure());
				pw.flush();
			}
			// what other measures.... + correct them appropriately
		}		 
		//*/
		
		/*
                Evaluator eval;
                Evaluation results;
		System.out.println("BR");
		BinaryRelevanceClassifier br = new BinaryRelevanceClassifier();
		br.setBaseClassifier(Classifier.makeCopy(baseClassifier));
		br.setNumLabels(numLabels);
		br.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(br, testData);
                results.getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
		System.out.println("HammingLoss : " + results.getExampleBased().hammingLoss());
		System.out.println("Precision   : " + results.getLabelBased().precision());
		System.out.println("Recall      : " + results.getLabelBased().recall());
		System.out.println("F1          : " + results.getLabelBased().fmeasure());
                //*/
                
                Evaluator evalLP;
                Evaluation resultsLP;
                System.out.println("LP");
		LabelPowersetClassifier lp = new LabelPowersetClassifier(Classifier.makeCopy(baseClassifier),numLabels);
		lp.buildClassifier(trainData);
		evalLP = new Evaluator();
		resultsLP = evalLP.evaluate(lp, testData);
                resultsLP.getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
		System.out.println("HammingLoss : " + resultsLP.getExampleBased().hammingLoss());
		System.out.println("Precision   : " + resultsLP.getLabelBased().precision());
		System.out.println("Recall      : " + resultsLP.getLabelBased().recall());
		System.out.println("F1          : " + resultsLP.getLabelBased().fmeasure());
		//*/
		
		/*
		FileReader fr = new FileReader("e:/greg/work/datasets/multilabel/yeast/yeast-train.arff");
		Instances data = new Instances(fr);
		
		SMO baseClassifier = new SMO();		
		RAKEL rsc = new RAKEL();
		rsc.setNumLabels(14);
		rsc.setBaseClassifier(baseClassifier);
		rsc.setSizeOfSubset(6);
		rsc.setNumModels(20);
		rsc.buildClassifier(data);
		
		FileReader frTest = new FileReader("e:/greg/work/datasets/multilabel/yeast/yeast-test.arff");
		Instances testData = new Instances(frTest);		
		
		Evaluator eval = new Evaluator();
		Evaluation results = eval.evaluate(rsc, testData);
		System.out.println("Hamming Loss: " + results.getExampleBased().hammingLoss());
		*/
		
		/*
		Log.addTarget(System.out);
		//testPt5Transformation();
		PrintStream ps = new PrintStream(new FileOutputStream("wmlcp.log", true));
		Log.addTarget(ps);
		Experiment exp = sais2007();
		exp.breakOnException = true;
		Run run = exp.runTweaked();
		run.printTo(System.out);
		run.printTo(ps);
		run.exportToCSV("e:/greg/work/research/projects/wmlc/exp/sais2007-R008.csv");
		ps.close();*/
	}
	
	@SuppressWarnings("unused")
	private static void yeast() throws Exception
	{
		Experiment experiment = new Experiment();
		
		//Base dir enables us to share experiments without without editing too many file paths
		experiment.baseDir = ":/datasets/";
		
		//A datasetreference never contains the dataset, just loads from file 
		//and returns the instances. We dont want to serialize the instances
		//along with the experiment.

		//Add as many datasets as wanted.
		//The last arg is the number of labels.
		experiment.dataSets.add(new DatasetReference("yeast-train.arff", "yeast-test.arff", 14));
		
		//Add as many multilabel classifiers as wanted
		experiment.addClassifier("BinaryRelevanceClassifier", null);
		experiment.addClassifier("PT5", null);
		experiment.addClassifier("SubsetClassifier", null);
		
		//Choose a set of evaluation types: simple, threshold or crossvalidation.
		experiment.evaluations.put(Evaluations.SIMPLE, null);
		
		//Threshold needs arguments, these are passed as an array of objects (start, step, numSteps)
		//experiment.evaluations.put(Evaluations.THRESHOLD, 
		//		new Object[]{new Double(0.0), new Double(0.1), new Integer(11)});
		
		//Choose which measures to include in the output
		experiment.measures.add(Measures.EXAMPLEBASED);
		experiment.measures.add(Measures.MACROLABEL);
		experiment.measures.add(Measures.MICROLABEL);
		experiment.measures.add(Measures.TIMINGS);
		
		//Choose base classifiers to use with each multilabel classifier
		experiment.addBaseClassifier("weka.classifiers.bayes.NaiveBayes", null);
		experiment.addBaseClassifier("weka.classifiers.trees.J48", null);
		experiment.saveAs("d:/yeast.exp");
		
		//run() performs an evaluation for every combination of dataset, 
		//classifier, baseclassifier and evaluationtype
		Run run  = experiment.run();
		run.exportToCSV("d:/yeast-02.csv");
		
		run.printTo(System.out);
		
		//Saving uses the same file name as previously.
		//Runs are included in the file.
		experiment.save();
		
		
		//Experiment exp = Experiment.loadFromFile("d:/yeast.exp");
		//exp.run();
		//exp.save();
	}

	@SuppressWarnings("unused")
	private static Experiment reuters01() throws Exception {
		Experiment experiment = new Experiment();
		experiment.name = "reuters01";
		experiment.baseDir = "d:/rcv1/";
		experiment.dataSets.add(new DatasetReference("rcv1-subset-train1-tiny.arff", "rcv1-subset-test1.arff", 101));
		experiment.addClassifier("BinaryRelevanceClassifier", null);
		experiment.addClassifier("PT5", null);
		//experiment.classifiers.add("SubsetClassifier");
		//experiment.evaluations.put(Evaluations.SIMPLE, null);
		experiment.evaluations.put(Evaluations.THRESHOLD, 
				new Object[]{new Double(0.0), new Double(0.1), new Integer(11)});
		experiment.measures.add(Measures.EXAMPLEBASED);
		experiment.measures.add(Measures.MACROLABEL);
		experiment.measures.add(Measures.MICROLABEL);
		experiment.measures.add(Measures.TIMINGS);
		//experiment.baseClassifiers.add("weka.classifiers.bayes.NaiveBayesMultinomial");
		experiment.addBaseClassifier("weka.classifiers.functions.SMO", null);
		return experiment;
	}	

	@SuppressWarnings("unused")
	private static void testPt5Transformation() throws Exception
	{
		//No news is good news. If we get through this method without exceptions
		//Then transformations are working fine.
		Instances instances = new Instances(new FileReader("d:/rcv1/rcv1-subset-train1-tiny.arff"));
		FlattenTrueLabelsClassifier pt5 = new FlattenTrueLabelsClassifier();
		pt5.setDebug(true);
		pt5.setNumLabels(101);
		Log.log("Start build");
		pt5.buildClassifier(instances);
		Evaluation[] evals = new Evaluator().evaluateOverThreshold(pt5, instances, 0.0, 0.1, 11);
		
		//Non sparse test
		instances = new Instances(new FileReader("d:/datasets/scene-train.arff"));
		pt5.setNumLabels(6);
		pt5.buildClassifier(instances);
		evals = new Evaluator().evaluateOverThreshold(pt5, instances, 0.0, 0.1, 11);
	}
	
	
	
	@SuppressWarnings("unused")
	private static Experiment sais2007() throws Exception {

		Experiment experiment = new Experiment();
		
		experiment.name = "Nearest Subset First Trial";
		experiment.baseDir = "e:/greg/work/datasets/multilabel/";
		//experiment.dataSets.add(new DatasetReference("tmc2007/tmc2007-500-train.arff", "tmc2007/tmc2007-500-test.arff", 22));
		//experiment.dataSets.add(new DatasetReference("rcv1-subset-train1.arff", "rcv1-subset-test1.arff", 101));
		//experiment.dataSets.add(new DatasetReference("mediamill-train1.arff", "mediamill-test1.arff", 101));
		//experiment.dataSets.add(new DatasetReference("scene/scene-train.arff", "scene/scene-test.arff", 6));
		//experiment.dataSets.add(new DatasetReference("yeast/yeast-train.arff", "yeast/yeast-test.arff", 14));
		experiment.dataSets.add(new DatasetReference("genbase/genbase10-train.arff", "genbase/genbase10-test.arff", 27));
		
		//With and without Nearest subset flag
		//OptionRange range = new OptionRange("--max-subset-distance", new String[]{"1", "2", "3", "4"});
		//for(String opt: range) experiment.addClassifier("BinaryRelevanceClassifier", "--nearest-subset " + opt);
		
		//Baseline
		experiment.addClassifier("BinaryRelevanceClassifier", "-D");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method greedy");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method prob --diff 1");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method prob --diff 2");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method prob --diff 3");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method prob --diff 4");
		experiment.addClassifier("BinaryRelevanceClassifier", "-D --nearest-subset-method prob --diff 5");
		experiment.addBaseClassifier(
				"weka.classifiers.functions.LibSVM", 
				"-Z");
		//experiment.addBaseClassifier("weka.classifiers.lazy.IBk", "-K 3 -W 0 -A \"weka.core.KDTree -A weka.core.EuclideanDistance -W 0.01 -L 40\"");
		

		experiment.evaluations.put(Evaluations.SIMPLE, null);
		//experiment.evaluations.put(Evaluations.CROSSVALIDATION, new Object[]{10});
		experiment.measures.add(Measures.EXAMPLEBASED);
		experiment.measures.add(Measures.MACROLABEL);
		experiment.measures.add(Measures.MICROLABEL);
		experiment.measures.add(Measures.TIMINGS);
		
		return experiment;
		
		/*

		//New feature, generated options
		OptionRange optionRange = new OptionRange("-C", 0.15, 0.05, 3); //double start, double step, int numSteps
		optionRange.combine(new OptionRange("-M", new String[]{"2", "3", "4"})); //enumerate passed strings 
		optionRange.combine(new OptionRange("-A")); //use once with and once without
		

		for(String optionLine : optionRange)
			;//experiment.addBaseClassifier("weka.classifiers.trees.J48", "-C 0.25 -M 2 " + optionLine);
		
		
		//experiment.addBaseClassifier("weka.classifiers.trees.J48", "-C 0.25 -M 2 -A");

		*/
		
		//arg -1 is Leave one out cross validation
		
		
		//experiment.evaluations.put(Evaluations.SPLITLABEL, null);
		//experiment.evaluations.put(Evaluations.THRESHOLD, 
		//		new Object[]{new Double(0.0), new Double(0.1), new Integer(11)});
		
	}	
	
	
	
}

