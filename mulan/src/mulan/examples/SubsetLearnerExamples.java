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
package mulan.examples;

import java.util.Arrays;
import java.util.List;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.meta.EnsembleOfSubsetLearners;
import mulan.classifier.meta.SubsetLearner;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.*;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 * A main class with examples for SubsetLearner, GreedyLabelClustering and
 * EnsembleOfSubsetLearners methods usage.
 *
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 2012.02.06
 */
public class SubsetLearnerExamples {

    /**
     * Executes this example
     *
     * @param args command-line arguments -path, -filestem,
     * e.g. -path dataset/ -filestem emotions
     * @throws Exception exceptions not caught
     */
    public static void main(String[] args) throws Exception {
        String path = Utils.getOption("path", args); // e.g. -path dataset/
        String filestem = Utils.getOption("filestem", args); // e.g. -filestem emotions
        System.out.println("Loading the training set");
        MultiLabelInstances train = new MultiLabelInstances(path + filestem + "-train.arff", path
                + filestem + ".xml");
        System.out.println("Loading the test set");
        MultiLabelInstances test = new MultiLabelInstances(path + filestem + "-test.arff", path
                + filestem + ".xml");

        /*
         * The usage of the following methods is demonstrated: "GreedyLabelClustering-U" - an example for
         * running SubsetLearner using GreedyLabelClustering algorithm along with Unconditional labels dependence identification.
         * "EnsembleOfSubsetLearners-U" - an example for running EnsembleOfSubsetLearners algorithm using Unconditional
         * labels dependence identification. "GreedyLabelClustering-C" - an example for running
         * SubsetLearner using GreedyLabelClustering algorithm along with Conditional labels dependence identification.
         * "EnsembleOfSubsetLearners-C" - an example for running EnsembleOfSubsetLearners algorithm using Conditional
         * labels dependence identification. "SubsetLearner" - an example for running SubsetLearner
         * algorithm "UnconditionalLDI" - an example for running the algorithm for Unconditional
         * labels dependence identification. "ConditionalLDI" - an example for running the algorithm
         * for Conditional labels dependence identification.
         */
        String[] methodsToCompare = {"GreedyLabelClustering-U", "EnsembleOfSubsetLearners-U",
            "GreedyLabelClustering-C", "EnsembleOfSubsetLearners-C", "SubsetLearner",
            "UnconditionalLDI", "ConditionalLDI"};
        Evaluator eval = new Evaluator();
        Evaluation results;
        long s1, s2, s3;
        long trainTime, testTime;

        for (String aMethodsToCompare : methodsToCompare) {
            if (aMethodsToCompare.equals("GreedyLabelClustering-U")) {
                System.out.println("\nStarting GreedyLabelClustering algorithm using Unconditional labels dependence identification");
                UnconditionalChiSquareIdentifier uncond = new UnconditionalChiSquareIdentifier();
                MultiLabelLearner lp = new LabelPowerset(new J48());
                GreedyLabelClustering clusterer = new GreedyLabelClustering(lp, new J48(), uncond);
                SubsetLearner learner = new SubsetLearner(clusterer, lp, new J48());
                learner.setUseCache(true); // use caching mechanism
                learner.setDebug(true);
                s1 = System.currentTimeMillis();
                learner.build(train);
                s2 = System.currentTimeMillis();
                results = eval.evaluate(learner, test, train);
                s3 = System.currentTimeMillis();
                trainTime = s2 - s1;
                testTime = s3 - s2;
                System.out.println(results.toCSV());
                System.out.println("Train time: " + trainTime + " Test time: " + testTime);
            }

            if (aMethodsToCompare.equals("GreedyLabelClustering-C")) {
                System.out.println("\nStarting GreedyLabelClustering algorithm using Conditional labels dependence identification");
                ConditionalDependenceIdentifier cond = new ConditionalDependenceIdentifier(
                        new J48());
                MultiLabelLearner lp = new LabelPowerset(new J48());
                GreedyLabelClustering clusterer = new GreedyLabelClustering(lp, new J48(), cond);
                SubsetLearner learner = new SubsetLearner(clusterer, lp, new J48());
                learner.setUseCache(true); // use caching mechanism
                learner.setDebug(true);
                s1 = System.currentTimeMillis();
                learner.build(train);
                s2 = System.currentTimeMillis();
                results = eval.evaluate(learner, test, train);
                s3 = System.currentTimeMillis();
                trainTime = s2 - s1;
                testTime = s3 - s2;
                System.out.println(results.toCSV());
                System.out.println("Train time: " + trainTime + " Test time: " + testTime);
            }

            if (aMethodsToCompare.equals("EnsembleOfSubsetLearners-U")) {
                System.out.println("\nStarting EnsembleOfSubsetLearners algorithm using Unconditional labels dependence identification");
                UnconditionalChiSquareIdentifier uncond = new UnconditionalChiSquareIdentifier();
                MultiLabelLearner lp = new LabelPowerset(new J48());
                EnsembleOfSubsetLearners learner = new EnsembleOfSubsetLearners(lp, new J48(),
                        uncond, 10);
                learner.setDebug(true);
                learner.setUseSubsetLearnerCache(true);
                s1 = System.currentTimeMillis();
                learner.build(train);
                s2 = System.currentTimeMillis();
                results = eval.evaluate(learner, test, train);
                s3 = System.currentTimeMillis();
                trainTime = s2 - s1;
                testTime = s3 - s2;
                System.out.println(results.toCSV());
                System.out.println("Train time: " + trainTime + " Test time: " + testTime);
            }

            if (aMethodsToCompare.equals("EnsembleOfSubsetLearners-C")) {
                System.out.println("\nStarting EnsembleOfSubsetLearners algorithm using Conditional labels dependence identification");
                ConditionalDependenceIdentifier cond = new ConditionalDependenceIdentifier(
                        new J48());
                MultiLabelLearner lp = new LabelPowerset(new J48());
                EnsembleOfSubsetLearners learner = new EnsembleOfSubsetLearners(lp, new J48(),
                        cond, 10);
                learner.setDebug(true);
                learner.setUseSubsetLearnerCache(true);
                learner.setSelectDiverseModels(false);
                // use strategy for selecting highly weighted ensemble partitions (without seeking
                // for diverse models)
                s1 = System.currentTimeMillis();
                learner.build(train);
                s2 = System.currentTimeMillis();
                System.out.println("Evaluation started. ");
                results = eval.evaluate(learner, test, train);
                s3 = System.currentTimeMillis();
                trainTime = s2 - s1;
                testTime = s3 - s2;
                System.out.println(results.toCSV());
                System.out.println("Train time: " + trainTime + " Test time: " + testTime);
            }

            if (aMethodsToCompare.equals("SubsetLearner")) {
                System.out.println("\nStarting SubsetLearner algorithm with random label set partition.");
                EnsembleOfSubsetLearners ensemble = new EnsembleOfSubsetLearners();
                List<int[][]> randomSet = ensemble.createRandomSets(train.getNumLabels(), 1);
                int[][] partition = randomSet.get(0);
                System.out.println("Random partition: "
                        + EnsembleOfSubsetLearners.partitionToString(partition));
                SubsetLearner learner = new SubsetLearner(partition, new J48());
                learner.setDebug(true);
                s1 = System.currentTimeMillis();
                learner.build(train);
                s2 = System.currentTimeMillis();
                results = eval.evaluate(learner, test, train);
                s3 = System.currentTimeMillis();
                trainTime = s2 - s1;
                testTime = s3 - s2;
                System.out.println(results.toCSV());
                System.out.println("Train time: " + trainTime + " Test time: " + testTime);
            }

            if (aMethodsToCompare.equals("UnconditionalLDI")) {
                System.out.println("\nStarting algorithm for Unconditional labels dependence identification.");
                UnconditionalChiSquareIdentifier uncond = new UnconditionalChiSquareIdentifier();
                s1 = System.currentTimeMillis();
                LabelsPair[] pairs = uncond.calculateDependence(train);
                s2 = System.currentTimeMillis();
                testTime = s2 - s1;
                System.out.println("Identified dependency scores of label pairs: \n"
                        + Arrays.toString(pairs));
                System.out.println("Computation time: " + testTime);
            }

            if (aMethodsToCompare.equals("ConditionalLDI")) {
                System.out.println("\nStarting algorithm for Conditional labels dependence identification.");
                ConditionalDependenceIdentifier cond = new ConditionalDependenceIdentifier(
                        new J48());
                s1 = System.currentTimeMillis();
                LabelsPair[] pairs = cond.calculateDependence(train);
                s2 = System.currentTimeMillis();
                testTime = s2 - s1;
                System.out.println("Identified dependency scores of label pairs: \n"
                        + Arrays.toString(pairs));
                System.out.println("Computation time: " + testTime);
            }
        }
    }
}
