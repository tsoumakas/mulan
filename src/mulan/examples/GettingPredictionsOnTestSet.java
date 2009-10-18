/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    GettingPredictionsOnTestSet.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
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
 * @author Grigorios Tsoumakas
 */
public class GettingPredictionsOnTestSet {

    public static void main(String[] args) throws Exception {
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
