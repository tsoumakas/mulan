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
 *    TrainTestExperiment.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.examples;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.IncludeLabelsClassifier;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiClassLearner;
import mulan.classifier.transformation.RAkEL;
import mulan.core.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.transformations.multiclass.Copy;
import mulan.transformations.multiclass.Ignore;
import mulan.transformations.multiclass.MultiClassTransformation;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 *
 * @author greg
 */
public class TrainTestExperiment {

    public static void main(String[] args) {
        String[] methodsToCompare = {"MC-Copy", "IncludeLabels","MC-Ignore","RAkEL", "LP", "CLR", "BR"};

        try {
            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);
            MultiLabelInstances train = new MultiLabelInstances(path + filestem + "-train.arff", path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(path + filestem + "-test.arff", path + filestem + ".xml");
            
            Evaluator eval = new Evaluator();
            Evaluation results;

            for (int i=0; i<methodsToCompare.length; i++) {

                if (methodsToCompare[i].equals("BR")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier brClassifier = new J48();
                    BinaryRelevance br = new BinaryRelevance(brClassifier);
                    br.build(train);
                    results = eval.evaluate(br, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("LP")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier lpBaseClassifier = new J48();
                    LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
                    lp.setDebug(true);
                    lp.build(train);
                    results = eval.evaluate(lp, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("CLR")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier clrClassifier = new J48();
                    CalibratedLabelRanking clr = new CalibratedLabelRanking(clrClassifier);
                    clr.setDebug(true);
                    clr.build(train);
                    results = eval.evaluate(clr, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("RAkEL")) {
                    System.out.println(methodsToCompare[i]);
                    MultiLabelLearner lp = new LabelPowerset(new J48());
                    RAkEL rakel = new RAkEL(lp);
                    rakel.setDebug(true);
                    rakel.build(train);
                    results = eval.evaluate(rakel, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("MC-Copy")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier mclClassifier = new J48();
                    MultiClassTransformation mcTrans = new Copy();
                    MultiClassLearner mcl = new MultiClassLearner(mclClassifier, mcTrans);
                    mcl.setDebug(true);
                    mcl.build(train);
                    results = eval.evaluate(mcl, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("MC-Ignore")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier mclClassifier = new J48();
                    MultiClassTransformation mcTrans = new Ignore();
                    MultiClassLearner mcl = new MultiClassLearner(mclClassifier, mcTrans);
                    mcl.build(train);
                    results = eval.evaluate(mcl, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("IncludeLabels")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier ilClassifier = new J48();
                    IncludeLabelsClassifier il = new IncludeLabelsClassifier(ilClassifier);
                    il.setDebug(true);
                    il.build(train);
                    results = eval.evaluate(il, test);
                    System.out.println(results.toString());
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
