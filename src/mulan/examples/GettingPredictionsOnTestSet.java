/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;

import mulan.classifier.neural.BPMLL;
import mulan.classifier.neural.LossMeasure;
import mulan.classifier.neural.MMPLearner;
import mulan.classifier.neural.MMPUpdateRuleType;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;


/**
 *
 * @author greg
 */
public class GettingPredictionsOnTestSet {
    
    public static void main(String[] args) throws Exception
    {
        String path = "d:/work/datasets/multilabel/yeast/";
        String trainfile = path + "yeast-train.arff";
        String testfile = path + "yeast-test.arff";
        int numLabels = 14;

        MultiLabelInstances traindata = new MultiLabelInstances(trainfile, numLabels);
        MultiLabelInstances testdata = new MultiLabelInstances(testfile, numLabels);
        Evaluator eval = new Evaluator(5);
        Evaluation results;
        
        //* Label Powerset Classifier
        System.out.println("LP");
        LabelPowerset lp = new LabelPowerset(new J48());
        lp.build(traindata);
        results = eval.evaluate(lp, testdata);
        System.out.println(results.toString());
        System.gc(); 
        //*/
        
        //* BPMLL Classifier
        System.out.println("BPMLL");
        BPMLL bpmll = new BPMLL();
        bpmll.setHiddenLayers(new int[]{50});
        bpmll.setDebug(true);
        bpmll.build(traindata);
        results = eval.evaluate(bpmll, testdata);
        System.out.println(results.toString());
        System.gc(); 
        //*/
        
        //* MMP Learner
        System.out.println("MMP");
        MMPLearner mmp = new MMPLearner(LossMeasure.OneError, MMPUpdateRuleType.UniformUpdate);
        mmp.setDebug(true);
        mmp.build(traindata);
        results = eval.evaluate(mmp, testdata);
        System.out.println(results.toString());
        System.gc(); 
        //*/
    }
}
