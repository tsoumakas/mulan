import java.util.Random;
import mulan.classifier.AbstractMultiLabelClassifier;
import mulan.evaluation.CrossValidation;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.ExampleBasedCrossValidation;
import mulan.evaluation.ExampleBasedEvaluation;
import mulan.evaluation.LabelBasedCrossValidation;
import mulan.evaluation.LabelBasedEvaluation;

import weka.classifiers.Classifier;
import weka.core.Instances;


/**
 * Custom evaluation class used to save time retraining classifiers.
 * 
 *
 */

public class InterlacedCV
{
	Fold[] folds = new Fold[10];
	
	Random random = new Random(1);
	class Fold
	{
		AbstractMultiLabelClassifier classifier;
		Instances test;
	}

	public InterlacedCV(AbstractMultiLabelClassifier classifier,
				Instances dataset, int numLabels)
	throws Exception
	{
		int numFolds = 10;
		Log.log("Randomizing complete dataset: " + dataset.relationName());
		dataset.randomize(random);
		
		for(int i = 0; i < numFolds; i++)
		{
			Log.log("Extracting training/test from fold " + i);
			Instances train = dataset.trainCV(numFolds, i, random);
			Instances test  = dataset.testCV(numFolds, i);
			folds[i] = new Fold();
			folds[i].classifier = (AbstractMultiLabelClassifier) Classifier.makeCopy(classifier);
			Log.log("Building fold classifier");
			folds[i].classifier.buildClassifier(train);
			folds[i].test = test;
		}
	}
	
	public void setNearestSubsetMethod(AbstractMultiLabelClassifier.SubsetMappingMethod mm)
	{
		for(Fold f: folds)
		{
			f.classifier.setSubsetMethod(mm);
		}
	}
	
	public Evaluation evaluate() throws Exception
	{
		Evaluator evaluator = new Evaluator();
		LabelBasedEvaluation[] labelBased = new LabelBasedEvaluation[10];
		ExampleBasedEvaluation[] exampleBased = new ExampleBasedEvaluation[10];
		
		for(int i = 0; i < folds.length; i++)
		{
			Fold f = folds[i];
			Log.log("Evaluating test set of fold " + i);
			Evaluation evaluation = evaluator.evaluate(f.classifier, f.test);
			labelBased[i] = evaluation.getLabelBased();
			exampleBased[i] = evaluation.getExampleBased();
		}

                return null;
		/* tofix 
                return new CrossValidation(
			new LabelBasedCrossValidation(labelBased),
			new ExampleBasedCrossValidation(exampleBased),
			10); 		
                 **/
	}
}
