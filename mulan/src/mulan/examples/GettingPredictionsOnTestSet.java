/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;

import java.io.FileReader;
import mulan.classifier.LabelPowerset;
import mulan.classifier.neural.BPMLL;
import mulan.evaluation.BinaryPrediction;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedEvaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author greg
 */
public class GettingPredictionsOnTestSet {
    
    public static void main(String[] args) throws Exception
    {
        String path = "d:/work/datasets/multilabel/yeast/";
        String trainfile = "yeast-train.arff";
        String testfile = "yeast-test.arff";
        int numLabels = 14;

        FileReader frtrainData = new FileReader(path + trainfile);
        Instances traindata = new Instances(frtrainData);                
        FileReader frtestData = new FileReader(path + testfile);
        Instances testdata = new Instances(frtestData);          
        Evaluator eval = new Evaluator(5);
        IntegratedEvaluation results;
        
        //* Label Powerset Classifier
        System.out.println("LP");
        J48 lpBaseClassifier = new J48();
        LabelPowerset lp = new LabelPowerset(lpBaseClassifier, numLabels);
        lp.buildClassifier(traindata);
        results = eval.evaluateAll(lp, testdata);
        BinaryPrediction[][] preds = results.getPredictions();
        for (int i=0; i<preds.length; i++)
        {
            System.out.print("test example " + (i+1) + ": ");                    
            for (int j=0; j<preds[i].length; j++)
            {
                boolean prediction = preds[i][j].getPrediction();
                if (prediction)
                    System.out.print(testdata.attribute(testdata.numAttributes()-numLabels+j).name() + " ");
            }
            System.out.println();
        }
        System.out.println(results.toString());
        System.gc(); 
        //*/
        
        //* BPMLL Classifier
        System.out.println("BPMLL");
        BPMLL bpmll = new BPMLL(numLabels);
        bpmll.setHiddenLayers(new int[]{50});
        bpmll.setDebug(true);
        bpmll.buildClassifier(traindata);
        results = eval.evaluateAll(bpmll, testdata);
        BinaryPrediction[][] predictions = results.getPredictions();
        for (int i=0; i<predictions.length; i++)
        {
            System.out.print("test example " + (i+1) + ": ");                    
            for (int j=0; j<predictions[i].length; j++)
            {
                boolean prediction = predictions[i][j].getPrediction();
                if (prediction)
                    System.out.print(testdata.attribute(testdata.numAttributes()-numLabels+j).name() + " ");
            }
            System.out.println();
        }
        System.out.println(results.toString());
        System.gc(); 
        //*/
    }
}
