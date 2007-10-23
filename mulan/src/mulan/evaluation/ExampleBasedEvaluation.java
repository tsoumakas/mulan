package mulan.evaluation;
/**
 * @author  rofr
 */
import weka.core.Utils;

public class ExampleBasedEvaluation extends EvaluationBase 
{

	protected double hammingLoss;
	protected double subsetAccuracy;
	protected double accuracy;
	protected double recall;
	protected double precision;
	protected double fmeasure;
	
	protected double forgivenessRate;

	
	protected ExampleBasedEvaluation(BinaryPrediction[][] predictions,
									 double forgivenessRate)
	{
		super(predictions);
		this.forgivenessRate = forgivenessRate;
		computeMeasures();
	}
	
	protected ExampleBasedEvaluation(BinaryPrediction[][] predictions)
	{
		this(predictions, 1.0);
	}
	
	protected ExampleBasedEvaluation(double forgivenessRate)
	{
		super(null);
		this.forgivenessRate = forgivenessRate;
	}


	protected void computeMeasures() //throws Exception
	{
		// Reset in case of multiple calls
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fmeasure = 0;
		subsetAccuracy = 0;

		int numLabels = numLabels();
		int numInstances = numInstances();

		for (int i = 0; i < numInstances; i++)
		{
			// Counter variables
			double setUnion = 0; // |Y or Z|
			double setIntersection = 0; // |Y and Z|
			double labelPredicted = 0; // |Z|
			double labelActual = 0; // |Y|
			double symmetricDifference = 0; // |Y xor Z|
			boolean setsIdentical = true; // innocent until proven guilty

			//Do the counting
			for (int j = 0; j < numLabels; j++)
			{
				boolean actual = predictions[i][j].actual;
				boolean predicted = predictions[i][j].predicted;

				if (predicted != actual)
				{
					symmetricDifference++;
					setsIdentical = false;
				}

				if (actual) labelActual++;
				if (predicted) labelPredicted++;

				if (predicted && actual) setIntersection++;
				if (predicted || actual) setUnion++;
			}

			if (setsIdentical) subsetAccuracy++;

			if(labelActual + labelPredicted == 0)
			{
				accuracy  += 1;
				recall    += 1;
				precision += 1;
				fmeasure  += 1;
			}
			else
			{
				if (Utils.eq(forgivenessRate, 1.0)) accuracy += (setIntersection / setUnion);
				else accuracy += Math.pow(setIntersection / setUnion, forgivenessRate);

				if (labelPredicted > 0) precision += (setIntersection / labelPredicted);
				if (labelActual > 0)    recall += (setIntersection / labelActual);
				
			}
			hammingLoss += (symmetricDifference / numLabels);

		}

		// Set final values
		hammingLoss /= numInstances;
		accuracy /= numInstances;
		precision /= numInstances;
		recall /= numInstances;
		subsetAccuracy /= numInstances;
		fmeasure = computeFMeasure(precision, recall);

	}

	public String toSummaryString()
	{
		// TODO: create and return a pretty string representation
		// summarizing all evaluation results

		return "Not implemented";
	}

	/**
	 * An additional measure specific to example based evaluation
	 * @return  the hammingLoss
	 */
	public double hammingLoss()
	{
		return hammingLoss;
	}


	/**
	 * An additional measure specific to example based evaluation
	 * @return  the subsetAccuracy
	 */
	public double subsetAccuracy()
	{
		return subsetAccuracy;
	}

	@Override
	public double accuracy()
	{
		return accuracy;
	}


	public double fmeasure()
	{
		return fmeasure;
	}

	@Override
	public double precision()
	{
		return precision;
	}

	@Override
	public double recall()
	{
		return recall;
	}
}
