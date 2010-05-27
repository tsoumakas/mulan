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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.examples;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HMC;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.IncludeLabelsClassifier;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiClassLearner;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.transformations.multiclass.Copy;
import mulan.transformations.multiclass.Ignore;
import mulan.transformations.multiclass.MultiClassTransformation;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 *
 * @author Grigorios Tsoumakas
 */
public class TrainTestExperiment {

    public static void main(String[] args) {
        String[] methodsToCompare = {"HOMER", "BR", "CLR", "MLkNN", "MC-Copy", "IncludeLabels", "MC-Ignore", "RAkEL", "LP", "MLStacking"};

        try {
            String path = Utils.getOption("path", args); // e.g. -path dataset/
            String filestem = Utils.getOption("filestem", args); // e.g. -filestem emotions
            String percentage = Utils.getOption("percentage", args); // e.g. -percentage 50 (for 50%)
            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");
            
            //split the data set into train and test
            Instances dataSet = mlDataSet.getDataSet();
            //dataSet.randomize(new Random(1));
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInvertSelection(true);
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);
            
            rmvp = new RemovePercentage();
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances testDataSet = Filter.useFilter(dataSet, rmvp);
            
            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            Evaluation results;

            for (int i = 0; i < methodsToCompare.length; i++) {

                if (methodsToCompare[i].equals("BR")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier brClassifier = new NaiveBayes();
                    BinaryRelevance br = new BinaryRelevance(brClassifier);
                    br.setDebug(true);
                    br.build(train);
                    results = eval.evaluate(br, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("LP")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier lpBaseClassifier = new J48();
                    LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
                    lp.setDebug(true);
                    lp.build(train);
                    results = eval.evaluate(lp, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("CLR")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier clrClassifier = new J48();
                    CalibratedLabelRanking clr = new CalibratedLabelRanking(clrClassifier);
                    clr.setDebug(true);
                    clr.build(train);
                    results = eval.evaluate(clr, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("RAkEL")) {
                    System.out.println(methodsToCompare[i]);
                    MultiLabelLearner lp = new LabelPowerset(new J48());
                    RAkEL rakel = new RAkEL(lp);
                    rakel.setDebug(true);
                    rakel.build(train);
                    results = eval.evaluate(rakel, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("MC-Copy")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier mclClassifier = new J48();
                    MultiClassTransformation mcTrans = new Copy();
                    MultiClassLearner mcl = new MultiClassLearner(mclClassifier, mcTrans);
                    mcl.setDebug(true);
                    mcl.build(train);
                    results = eval.evaluate(mcl, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("MC-Ignore")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier mclClassifier = new J48();
                    MultiClassTransformation mcTrans = new Ignore();
                    MultiClassLearner mcl = new MultiClassLearner(mclClassifier, mcTrans);
                    mcl.build(train);
                    results = eval.evaluate(mcl, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("IncludeLabels")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier ilClassifier = new J48();
                    IncludeLabelsClassifier il = new IncludeLabelsClassifier(ilClassifier);
                    il.setDebug(true);
                    il.build(train);
                    results = eval.evaluate(il, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("MLkNN")) {
                    System.out.println(methodsToCompare[i]);
                    int numOfNeighbors = 10;
                    double smooth = 1.0;
                    MLkNN mlknn = new MLkNN(numOfNeighbors, smooth);
                    mlknn.setDebug(true);
                    mlknn.build(train);
                    results = eval.evaluate(mlknn, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("HMC")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier baseClassifier = new J48();
                    LabelPowerset lp = new LabelPowerset(baseClassifier);
                    RAkEL rakel = new RAkEL(lp);
                    HMC hmc = new HMC(rakel);
                    hmc.build(train);
                    results = eval.evaluate(hmc, test);
                    System.out.println(results);
                }

                if (methodsToCompare[i].equals("HOMER")) {
                    System.out.println(methodsToCompare[i]);
                    Classifier baseClassifier = new SMO();
                    CalibratedLabelRanking learner = new CalibratedLabelRanking(baseClassifier);
                    learner.setDebug(true);
                    HOMER homer = new HOMER(learner, 3, HierarchyBuilder.Method.Random);
                    homer.setDebug(true);
                    homer.build(train);
                    results = eval.evaluate(homer, test);
                    System.out.println(results);
                }
                if (methodsToCompare[i].equals("MLStacking")) {
                    System.out.println(methodsToCompare[i]);
                    int numOfNeighbors = 10;
                    Classifier baseClassifier = new IBk(numOfNeighbors);
                    Classifier metaClassifier = new Logistic();
                    MultiLabelStacking mls = new MultiLabelStacking(
                            baseClassifier, metaClassifier);
                    mls.setMetaPercentage(1.0);
                    mls.setDebug(true);
                    mls.build(train);
                    results = eval.evaluate(mls, test);
                    System.out.println(results);
                }

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
