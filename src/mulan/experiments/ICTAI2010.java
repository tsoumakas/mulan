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
 *    ICTAI2010.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.experiments;

/**
 * @author Grigorios Tsoumakas
 */
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.thresholding.MetaLabeler;
import mulan.classifier.meta.thresholding.OneThreshold;
import mulan.classifier.meta.thresholding.RCut;
import mulan.classifier.meta.thresholding.SCut;
import mulan.classifier.meta.thresholding.ThresholdPrediction;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class ICTAI2010 extends Experiment {

    public static void main(String[] args) {

        try {
            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);

            System.out.println("Loading the data set");
            MultiLabelInstances dataset = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            MultipleEvaluation results;
            List<Measure> measures = new ArrayList<Measure>(1);
            measures.add(new HammingLoss());

            int numFolds = 10;


            MultiLabelLearner[] learner = new MultiLabelLearner[4];
            String[] learnerName = new String[learner.length];

            learner[0] = new MLkNN(10, 1.0);
            learnerName[0] = "MLkNN";
            learner[1] = new CalibratedLabelRanking(new J48());
            learnerName[1] = "CLR";
            Bagging bagging = new Bagging();
            bagging.setClassifier(new J48());
            learner[2] = new BinaryRelevance(bagging);
            learnerName[2] = "BR";
            learner[3] = new BPMLL();
            learnerName[3] = "BPMLL";

            // loop over learners
            for (int i = 0; i < learner.length; i++) {
                // Default
                results = eval.crossValidate(learner[i].makeCopy(), dataset, measures, numFolds);
                System.out.println(learnerName[i]+";default;-;" + results.toCSV());

                // One Threshold
                OneThreshold ot;
                ot = new OneThreshold(learner[i].makeCopy(), new HammingLoss());
                results = eval.crossValidate(ot, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";one threshold;train;" + results.toCSV());
                ot = new OneThreshold(learner[i].makeCopy(), new HammingLoss(), 5);
                results = eval.crossValidate(ot, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";one threshold;5-cv;" + results.toCSV());

                // RCut
                RCut rcut;
                rcut = new RCut(learner[i].makeCopy());
                results = eval.crossValidate(rcut, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";rcut;cardinality;" + results.toCSV());
                rcut = new RCut(learner[i].makeCopy(), new HammingLoss());
                results = eval.crossValidate(rcut, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";rcut;train;" + results.toCSV());
                rcut = new RCut(learner[i].makeCopy(), new HammingLoss(), 5);
                results = eval.crossValidate(rcut, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";rcut;5-cv;" + results.toCSV());

                // SCut
                SCut scut;
                scut = new SCut(learner[i].makeCopy(), new HammingLoss());
                results = eval.crossValidate(scut, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";scut;train;" + results.toCSV());
                scut = new SCut(learner[i].makeCopy(), new HammingLoss(), 5);
                results = eval.crossValidate(scut, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";scut;5-cv;" + results.toCSV());

                // MetaLabeler
                MetaLabeler ml;
                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 1);
                ml.chooseContentBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;train;content;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 1);
                ml.chooseScoreBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;train;scores;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 1);
                ml.chooseRankBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;train;ranks;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 1);
                ml.chooseContentBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;train;content;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 1);
                ml.chooseScoreBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;train;scores;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 1);
                ml.chooseRankBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;train;ranks;" + results.toCSV());

                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 5);
                ml.chooseContentBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;5-cv;content;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 5);
                ml.chooseScoreBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;5-cv;scores;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new M5P(), 5);
                ml.chooseRankBased();
                ml.chooseNumericClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;m5p;5-cv;ranks;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 5);
                ml.chooseContentBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;5-cv;content;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 5);
                ml.chooseScoreBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;5-cv;scores;" + results.toCSV());
                ml = new MetaLabeler(learner[i].makeCopy(), new J48(), 5);
                ml.chooseRankBased();
                ml.chooseNominalClass();
                results = eval.crossValidate(ml, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";metalabeler;j48;5-cv;ranks;" + results.toCSV());

                // ThresholdPrediction
                ThresholdPrediction tp;
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 1);
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;train;content;" + results.toCSV());
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 1);
                tp.chooseScoreBased();
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;train;scores;" + results.toCSV());
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 1);
                tp.chooseRankBased();
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;train;ranks;" + results.toCSV());
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 5);
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;5-cv;content;" + results.toCSV());
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 5);
                tp.chooseScoreBased();
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;5-cv;scores;" + results.toCSV());
                tp = new ThresholdPrediction(learner[i].makeCopy(), new M5P(), 5);
                tp.chooseRankBased();
                results = eval.crossValidate(tp, dataset, measures, numFolds);
                System.out.println(learnerName[i]+";tp;m5p;5-cv;ranks;" + results.toCSV());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Marios Ioannou and George Sakkas and Grigorios Tsoumakas and Ioannis Vlahavas");
        result.setValue(Field.TITLE, "Obtaining Bipartitions from Score Vectors for Multi-Label Classification");
        result.setValue(Field.BOOKTITLE, "Proceedings of the 22th IEEE International Conference on Tools with Artificial Intelligence");
        result.setValue(Field.YEAR, "2010");
        result.setValue(Field.PUBLISHER, "IEEE");
        return result;
    }
}
