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
 *    ICDM08EnsembleOfPrunedSets.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.experiments;

/**
 * @author Emmanouela Stachtiari
 * @author Grigorios Tsoumakas
 * @version 2010.12.10
 */
import java.util.HashMap;
import java.util.LinkedList;

import java.util.Random;
import mulan.classifier.meta.thresholding.OneThreshold;
import mulan.classifier.transformation.EnsembleOfPrunedSets;
import mulan.classifier.transformation.PrunedSets;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.BipartitionMeasureBase;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Class replicating an experiment from a published paper
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.27
 */
public class ICDM08EnsembleOfPrunedSets extends Experiment {

    /**
     * Main class
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {

        try {
            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);

            System.out.println("Loading the data set");
            MultiLabelInstances dataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            Evaluator evaluator;

            Measure[] evaluationMeasures = new Measure[2];
            evaluationMeasures[0] = new ExampleBasedAccuracy(false);
            evaluationMeasures[1] = new HammingLoss();
            evaluationMeasures[2] = new ExampleBasedFMeasure(false);

            HashMap<String, MultipleEvaluation> result = new HashMap<String, MultipleEvaluation>();
            for (Measure m : evaluationMeasures) {
                MultipleEvaluation me = new MultipleEvaluation();
                result.put(m.getName(), me);
            }

            Random random = new Random(1);

            for (int repetition = 0; repetition < 5; repetition++) {
                // perform 2-fold CV and add each to the current results
                dataSet.getDataSet().randomize(random);
                for (int fold = 0; fold < 2; fold++) {
                    System.out.println("Experiment " + (repetition * 2 + fold + 1));
                    Instances train = dataSet.getDataSet().trainCV(2, fold);
                    MultiLabelInstances multiTrain = new MultiLabelInstances(train, dataSet.getLabelsMetaData());
                    Instances test = dataSet.getDataSet().testCV(2, fold);
                    MultiLabelInstances multiTest = new MultiLabelInstances(test, dataSet.getLabelsMetaData());

                    HashMap<String, Integer> bestP = new HashMap<String, Integer>();
                    HashMap<String, Integer> bestB = new HashMap<String, Integer>();
                    HashMap<String, PrunedSets.Strategy> bestStrategy = new HashMap<String, PrunedSets.Strategy>();
                    HashMap<String, Double> bestDiff = new HashMap<String, Double>();
                    for (Measure m : evaluationMeasures) {
                        bestDiff.put(m.getName(), Double.MAX_VALUE);
                    }

                    System.out.println("Searching parameters");
                    for (int p = 5; p > 1; p--) {
                        for (int b = 1; b < 4; b++) {
                            MultipleEvaluation innerResult = null;
                            LinkedList<Measure> measures;
                            PrunedSets ps;
                            double diff;

                            evaluator = new Evaluator();
                            ps = new PrunedSets(new SMO(), p, PrunedSets.Strategy.A, b);
                            measures = new LinkedList<Measure>();
                            for (Measure m : evaluationMeasures) {
                                measures.add(m.makeCopy());
                            }
                            System.out.print("p=" + p + " b=" + b + " strategy=A ");
                            innerResult = evaluator.crossValidate(ps, multiTrain, measures, 5);
                            for (Measure m : evaluationMeasures) {
                                System.out.print(m.getName() + ": " + innerResult.getMean(m.getName()) + " ");
                                diff = Math.abs(m.getIdealValue() - innerResult.getMean(m.getName()));
                                if (diff <= bestDiff.get(m.getName())) {
                                    bestDiff.put(m.getName(), diff);
                                    bestP.put(m.getName(), p);
                                    bestB.put(m.getName(), b);
                                    bestStrategy.put(m.getName(), PrunedSets.Strategy.A);
                                }
                            }
                            System.out.println();

                            evaluator = new Evaluator();
                            ps = new PrunedSets(new SMO(), p, PrunedSets.Strategy.B, b);
                            measures = new LinkedList<Measure>();
                            for (Measure m : evaluationMeasures) {
                                measures.add(m.makeCopy());
                            }
                            System.out.print("p=" + p + " b=" + b + " strategy=B ");
                            innerResult = evaluator.crossValidate(ps, multiTrain, measures, 5);
                            for (Measure m : evaluationMeasures) {
                                System.out.print(m.getName() + ": " + innerResult.getMean(m.getName()) + " ");
                                diff = Math.abs(m.getIdealValue() - innerResult.getMean(m.getName()));
                                if (diff <= bestDiff.get(m.getName())) {
                                    bestDiff.put(m.getName(), diff);
                                    bestP.put(m.getName(), p);
                                    bestB.put(m.getName(), b);
                                    bestStrategy.put(m.getName(), PrunedSets.Strategy.B);
                                }
                            }
                            System.out.println();
                        }
                    }

                    for (Measure m : evaluationMeasures) {
                        System.out.println(m.getName());
                        System.out.println("Best p: " + bestP.get(m.getName()));
                        System.out.println("Best strategy: " + bestStrategy.get(m.getName()));
                        System.out.println("Best b: " + bestB.get(m.getName()));
                        EnsembleOfPrunedSets eps = new EnsembleOfPrunedSets(63, 10, 0.5, bestP.get(m.getName()), bestStrategy.get(m.getName()), bestB.get(m.getName()), new SMO());
                        OneThreshold ot = new OneThreshold(eps, (BipartitionMeasureBase) m.makeCopy(), 5);
                        ot.build(multiTrain);
                        System.out.println("Best threshold: " + ot.getThreshold());
                        evaluator = new Evaluator();
                        Evaluation e = evaluator.evaluate(ot, multiTest);
                        System.out.println(e.toCSV());
                        result.get(m.getName()).addEvaluation(e);
                    }
                }
            }
            for (Measure m : evaluationMeasures) {
                System.out.println(m.getName());
                result.get(m.getName()).calculateStatistics();
                System.out.println(result.get(m.getName()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.CONFERENCE);
        result.setValue(Field.AUTHOR, "Read, Jesse");
        result.setValue(Field.TITLE, "Multi-label Classification using Ensembles of Pruned Sets");
        result.setValue(Field.PAGES, "995-1000");
        result.setValue(Field.BOOKTITLE, "ICDM'08: Eighth IEEE International Conference on Data Mining");
        result.setValue(Field.YEAR, "2008");
        return result;
    }
}
