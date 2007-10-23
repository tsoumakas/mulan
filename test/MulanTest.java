/*
 * MulanTest.java
 *
 * Created on 30 Απρίλιος 2007, 4:34 μμ
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

/**
 *
 * @author greg
 */

import mulan.classifier.BinaryRelevanceClassifier;
import mulan.classifier.AbstractMultiLabelClassifier.*;
import mulan.classifier.IncludeLabelsClassifier;
import mulan.classifier.FlattenTrueLabelsClassifier;
import mulan.classifier.LabelPowersetClassifier;
import mulan.classifier.RecursiveLabelClustering;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.LabelBasedEvaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.Classifier;
import java.io.*;
import mulan.*;

public class MulanTest {
    
    /**
     * Creates a new instance of MulanTest
     */
    public MulanTest() {
    }
    
	public static void main(String[] args) throws Exception
	{
                /*
		String path = "c:/work/datasets/multilabel/mediamill/";
		String datastem = "mediamill1";
		int numLabels = 101;
                //*/
            
                /*
		String path = "c:/work/datasets/multilabel/tmc2007/";
		String datastem = "tmc2007";
		int numLabels = 22;
                //*/
            
		/*
                String path = "c:/work/datasets/multilabel/yeast/";
		String datastem = "yeast";
		int numLabels = 14;
                //*/
            
                //*
                String path = "c:/work/datasets/delicious/";
                String datastem = "d50-f2000";
                int numLabels = 983;
                //*/ 
                
                /*
                String path = "c:/work/datasets/multilabel/scene/";
		String datastem = "scene";
		int numLabels = 6;
                //*/
		
		FileReader frTrain = new FileReader(path + datastem + "-train.arff");
		Instances trainData = new Instances(frTrain);
		FileReader frTest = new FileReader(path + datastem + "-test.arff");
		Instances testData = new Instances(frTest);

                
                /* show multilabel statistics
		Instances allData = new Instances(trainData);
                for (int i=0; i<testData.numInstances(); i++)
                    allData.add(testData.instance(i));

                Statistics stats = new Statistics();
                stats.calculateStats(allData, numLabels);
                System.out.println(stats.toString());
                //*/
                
                // Define evaluation classes
                Evaluator eval;
                Evaluation results;
                
                //* base classifier: SMO                                     
                SMO baseClassifier = new SMO();
                /* PolyKernel 3
                PolyKernel pk = new PolyKernel();
                pk.setExponent(3);
                baseClassifier.setKernel(pk);
                //*/

                /* base classifier: NB                                      
                NaiveBayes baseClassifier = new NaiveBayes();
                //*/
                
                //* Recursive Label Clustering Classifier
		System.out.println("RLC");
                RecursiveLabelClustering rlc = new RecursiveLabelClustering();
                rlc.setNumLabels(numLabels);
                rlc.setMaxElements(10);
                rlc.setNumClusters(10);
                rlc.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(rlc, testData);
                System.out.println(results.toString());
                System.gc();
                //*/
                

                /* Binary Relevance Classifier
		System.out.println("BR");
		BinaryRelevanceClassifier br = new BinaryRelevanceClassifier();
		br.setBaseClassifier(Classifier.makeCopy(baseClassifier));
		br.setNumLabels(numLabels);
		br.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(br, testData);
                System.out.println(results.toString());
                System.gc();
                //*/
                               
                /* Label Powerset Classifier
                System.out.println("LP");
		LabelPowersetClassifier lp = new LabelPowersetClassifier(Classifier.makeCopy(baseClassifier), numLabels);
		lp.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(lp, testData);
                System.out.println(results.toString());
                System.gc();
		//*/

                /* BRmap1                
		System.out.println("BRmap1");
		BinaryRelevanceClassifier brMap1 = new BinaryRelevanceClassifier();
                brMap1.setSubsetMethod(SubsetMappingMethod.GREEDY);
		br.setBaseClassifier(Classifier.makeCopy(baseClassifier));
		br.setNumLabels(numLabels);
		br.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(br, testData);
                System.out.println(results.toString());
                System.gc();
                //*/
                
                /* Label Inclusion Classifier
                System.out.println("IL");
		IncludeLabelsClassifier il = new IncludeLabelsClassifier(Classifier.makeCopy(baseClassifier), numLabels);
		il.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(il, testData);
                System.out.println(results.toString());
                System.gc();
		//*/

                /* Flatten True Labels Classifier
                System.out.println("FTL");
		FlattenTrueLabelsClassifier ftl = new FlattenTrueLabelsClassifier(Classifier.makeCopy(baseClassifier), numLabels);
		ftl.buildClassifier(trainData);
		eval = new Evaluator();
		results = eval.evaluate(ftl, testData);
                System.out.println(results.toString());
                System.gc();
		//*/
        


        }    
}
