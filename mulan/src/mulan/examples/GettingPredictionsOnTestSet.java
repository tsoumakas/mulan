/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;

import java.io.FileReader;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
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
        Evaluation results;
        
        //* Label Powerset Classifier
        System.out.println("LP");
        LabelPowerset lp = new LabelPowerset(new J48(),numLabels);
        lp.build(traindata);
        results = eval.evaluate(lp, testdata);
        System.out.println(results.toString());
        System.gc(); 
        //*/
        
        //* BPMLL Classifier
        System.out.println("BPMLL");
        BPMLL bpmll = new BPMLL(numLabels);
        bpmll.setHiddenLayers(new int[]{50});
        bpmll.setDebug(true);
        bpmll.build(traindata);
        results = eval.evaluate(bpmll, testdata);
        System.out.println(results.toString());
        System.gc(); 
        //*/
    }
}
