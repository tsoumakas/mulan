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
package mulan.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.lazy.IBLR_ML;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Utils;

/**
 * <p>Class replicating the experiment in <em>Weiwei Cheng and Eyke Hüllermeier. 
 * 2009. Combining instance-based learning and logistic regression for 
 * multilabel classification. Mach. Learn. 76, 2-3 (September 2009), 211-225.
 * </em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis (espyromi@csd.auth.gr)
 * @version 2010.12.10
 */
public class MachineLearning09IBLR {

    /**
     * Main code
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

            List<Measure> measures = new ArrayList<>(5);
            measures.add(new HammingLoss());
            measures.add(new OneError());
            measures.add(new Coverage());
            measures.add(new RankingLoss());
            measures.add(new AveragePrecision());

            MultipleEvaluation iblrmlResults = new MultipleEvaluation(dataSet);
            MultipleEvaluation iblrmlPlusResults = new MultipleEvaluation(dataSet);

            Random random = new Random(1);

            for (int repetition = 0; repetition < 10; repetition++) {
                // perform 10-fold CV and add each to the current results
                dataSet.getDataSet().randomize(random);
                for (int fold = 0; fold < 10; fold++) {
                    System.out.println("Experiment " + (repetition * 10 + fold + 1));
                    Instances train = dataSet.getDataSet().trainCV(10, fold);
                    MultiLabelInstances multiTrain = new MultiLabelInstances(
                            train, dataSet.getLabelsMetaData());
                    Instances test = dataSet.getDataSet().testCV(10, fold);
                    MultiLabelInstances multiTest = new MultiLabelInstances(
                            test, dataSet.getLabelsMetaData());

                    System.out.println("IBLR-ML Experiment");
                    IBLR_ML iblrml = new IBLR_ML();
                    iblrml.build(multiTrain);
                    evaluator = new Evaluator();
                    Evaluation e1 = evaluator.evaluate(iblrml, multiTest, measures);
                    System.out.println(e1.toCSV());
                    iblrmlResults.addEvaluation(e1);

                    /*
                     The following code produces the same results, as IBLR
                     is equivalent to stacking using kNN at the 1st level
                     and Logistic Regression at the 2nd level
                     */
                    System.out.println("ML-Stacking Experiment");
                    int numOfNeighbors = 10;
                    Classifier baseClassifier = new IBk(numOfNeighbors);
                    Classifier metaClassifier = new Logistic();
                    MultiLabelStacking mls = new MultiLabelStacking(baseClassifier, metaClassifier);
                    mls.setMetaPercentage(1.0);
                    mls.build(multiTrain);
                    evaluator = new Evaluator();
                    Evaluation e1b = evaluator.evaluate(mls, multiTest, measures);
                    System.out.println(e1b.toCSV());
                    iblrmlResults.addEvaluation(e1b);
                    //*/

                    System.out.println("IBLR-ML+ Experiment");
                    IBLR_ML iblrmlplus = new IBLR_ML(10, true);
                    iblrmlplus.build(multiTrain);
                    evaluator = new Evaluator();
                    Evaluation e2 = evaluator.evaluate(iblrmlplus, multiTest, measures);
                    System.out.println(e2.toCSV());
                    iblrmlPlusResults.addEvaluation(e2);
                }

            }

            iblrmlResults.calculateStatistics();
            System.out.println(iblrmlResults);

            iblrmlPlusResults.calculateStatistics();
            System.out.println(iblrmlPlusResults);
        } catch (Exception ex) {
            Logger.getLogger(MachineLearning09IBLR.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}