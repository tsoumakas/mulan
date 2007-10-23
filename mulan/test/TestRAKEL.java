/*
 * TestRAKEL.java
 *
 * Created on 17 Ιούνιος 2007, 11:04 πμ
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.

 */
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import mulan.classifier.RAKEL;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.LabelBasedEvaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author greg
 */
public class TestRAKEL {
    
    public TestRAKEL() {  }

    private static int binomial(int n, int m) 
    {
        int[] b = new int[n+1];
        b[0]=1;
        for (int i=1; i<=n; i++)
        {
            b[i] = 1;
            for (int j=i-1; j>0; --j)
                b[j] += b[j-1];
        }
        return b[m];
    }    

    public static void main(String[] args) throws Exception
    {
        // dataset
        /*
        String path = "c:/work/datasets/multilabel/scene/";
	String datastem = "scene";
	int numLabels = 6;
        //*/
        /*
        String path = "c:/work/datasets/multilabel/yeast/";
        String datastem = "yeast";
        int numLabels = 14;
        int minK=7;
        int maxK=7; //numLabels;
        int stepK=1;
        int maxM=300; 
        //*/
        //*
        String path = "e:/greg/work/datasets/multilabel/tmc2007/";
        String datastem = "tmc2007-500";
        int numLabels = 22;
        int minK=5;
        int maxK=7; //numLabels;
        int stepK=2;
        int maxM = 50;
        int numFolds=2;
        //*/
        
        FileReader frTrain = new FileReader(path + datastem + "-train.arff");
	Instances trainData = new Instances(frTrain);
	FileReader frTest = new FileReader(path + datastem + "-test.arff");
	Instances testData = new Instances(frTest);
        
        // base classifier, default linear kernel, default C=1
        //SMO baseClassifier = new SMO();  
        J48 baseClassifier = new J48();
        
        BufferedWriter bw = new BufferedWriter(new FileWriter("results-" + datastem + ".txt"));
             
        //* Evaluate using X-fold CV
        for (int f=0; f<numFolds; f++)
        {            
            Instances foldTrainData = trainData.trainCV(numFolds, f);
            Instances foldTestData = trainData.testCV(numFolds, f);
            
            // rakel    
            for (int k=minK; k<=maxK; k+=stepK)
            {            
                RAKEL rakel = new RAKEL(numLabels,binomial(numLabels, k), k);
                rakel.setBaseClassifier(baseClassifier);
                int finalM = Math.min(binomial(numLabels,k),maxM);
                for (int m=0; m<finalM; m++)
                {
                    rakel.updateClassifier(foldTrainData, m);
                    Evaluator evaluator = new Evaluator();
                    rakel.updatePredictions(foldTestData, m);
                    rakel.nullSubsetClassifier(m);
                    Evaluation[] results = evaluator.evaluateOverThreshold(rakel.getPredictions(), foldTestData, 0.1, 0.1, 9);
                    for (int t=0; t<results.length; t++) 
                    {
                        results[t].getLabelBased().setAveragingMethod(LabelBasedEvaluation.MICRO);
                        bw.write("fold=" + f + ";k=" + k + 
                                           ";model=" + m + ";t=0." + (t+1) + 
                                            ";hl=" + results[t].getExampleBased().hammingLoss() +
                                            ";pr=" + results[t].getLabelBased().precision() + 
                                            ";re=" + results[t].getLabelBased().recall() +
                                            ";f1=" + results[t].getLabelBased().fmeasure() + "\n");
                        /*
                        System.out.println("fold=" + f + ";k=" + k + 
                                           ";model=" + m + ";t=0." + (t+1) + 
                                            ";hl=" + results[t].getExampleBased().hammingLoss() +
                                            ";pr=" + results[t].getLabelBased().precision() + 
                                            ";re=" + results[t].getLabelBased().recall() +
                                            ";f1=" + results[t].getLabelBased().fmeasure());
                         //*/
                    }  

                }
            }
        }
        bw.close();
        //*/
        
        //rakel.cvParameterSelection(trainData, 2);        
    }
}
