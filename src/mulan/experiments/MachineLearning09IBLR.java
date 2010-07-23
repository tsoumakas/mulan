package mulan.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mulan.classifier.lazy.IBLR_ML;
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
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class MachineLearning09IBLR extends Experiment{

	public static void main(String[] args) {

		try {
			String path = Utils.getOption("path", args);
			String filestem = Utils.getOption("filestem", args);

			System.out.println("Loading the data set");
			MultiLabelInstances dataSet = new MultiLabelInstances(path
					+ filestem + ".arff", path + filestem + ".xml");

			Evaluator evaluator = new Evaluator();

			List<Measure> measures = new ArrayList<Measure>(5);
			measures.add(new HammingLoss());
			measures.add(new OneError());
			measures.add(new Coverage());
			measures.add(new RankingLoss());
			measures.add(new AveragePrecision());

			MultipleEvaluation iblrmlResults = new MultipleEvaluation();
			MultipleEvaluation iblrmlPlusResults = new MultipleEvaluation();

			Random random = new Random(1);

			for (int repetition = 0; repetition < 10; repetition++) {
				// perform 10-fold CV and add each to the current results
				dataSet.getDataSet().randomize(random);
				for (int fold = 0; fold < 10; fold++) {
					System.out.println("Experiment "
							+ (repetition * 10 + fold + 1));
					Instances train = dataSet.getDataSet().trainCV(10, fold);
					MultiLabelInstances multiTrain = new MultiLabelInstances(
							train, dataSet.getLabelsMetaData());
					Instances test = dataSet.getDataSet().testCV(10, fold);
					MultiLabelInstances multiTest = new MultiLabelInstances(
							test, dataSet.getLabelsMetaData());

					System.out.println("IBLR-ML Experiment");
					IBLR_ML iblrml = new IBLR_ML();
					// iblrml.setDontNormalize(true);
					iblrml.build(multiTrain);
					evaluator = new Evaluator();
					Evaluation e1 = evaluator.evaluate(iblrml, multiTest,
							measures);
					System.out.println(e1.toCSV());
					iblrmlResults.addEvaluation(e1);

					/*
					 * The following code produces the same results, as IBLR 
					 * is equivalent to stacking using kNN at the 1st level
					 * and Logistic Regression at the 2nd level
					 * 
					 * System.out.println("ML-Stacking Experiment");
					 * int numOfNeighbors = 10; 
					 * Classifier baseClassifier = new IBk(numOfNeighbors); 
					 * Classifier metaClassifier = new Logistic();
					 * MultiLabelStacking mls = new MultiLabelStacking( baseClassifier, metaClassifier);
					 * mls.setMetaPercentage(1.0); 
					 * mls.build(multiTrain);
					 * evaluator = new Evaluator(); 
					 * Evaluation e1 = evaluator.evaluate(mls, multiTest, measures);
					 * System.out.println(e1.toCSV());
					 * iblrmlResults.addEvaluation(e1);
					 */

					System.out.println("IBLR-ML+ Experiment");
					IBLR_ML iblrmlplus = new IBLR_ML();
					iblrmlplus.setAddFeatures(true);
					// iblrmlplus.setDontNormalize(true);
					iblrmlplus.build(multiTrain);
					evaluator = new Evaluator();
					Evaluation e2 = evaluator.evaluate(iblrmlplus, multiTest,
							measures);
					System.out.println(e2.toCSV());
					iblrmlPlusResults.addEvaluation(e2);
				}

			}

			iblrmlResults.calculateStatistics();
			System.out.println(iblrmlResults);

			iblrmlPlusResults.calculateStatistics();
			System.out.println(iblrmlPlusResults);

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

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Weiwei Cheng and Eyke Hullermeier");
        result.setValue(
                Field.TITLE,
                "Combining instance-based learning and logistic regression for multilabel classification ");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "76");
        result.setValue(Field.NUMBER, "2-3");
        result.setValue(Field.YEAR, "2009");
        result.setValue(Field.ISSN, "0885-6125");
        result.setValue(Field.PAGES, "211-225");
        result.setValue(Field.PUBLISHER, "Springer Netherlands");

        return result;
    }
}
