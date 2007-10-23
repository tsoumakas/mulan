package mulan.evaluation;
import weka.core.Utils;

/**
 * Common base class for concrete Evaluation classes.  
 *
 */
abstract class EvaluationBase
{
	/**
	 * This is all the information needed to derive
	 * the measures and curves.
	 */
	protected BinaryPrediction[][] predictions;
		
	/**
	 * Constructor
	 * @param predictions The predictions to used to calculate measures
	 */
	protected EvaluationBase(BinaryPrediction[][] predictions)
	{
		this.predictions = predictions;
	}

	/**
	 * @return size of the testset. (total number of predictions)
	 */
	public int numInstances()
	{
		return predictions.length;
	}

	/**
	 * 
	 * @return total number of possible labels
	 */
	public int numLabels()
	{
		return predictions[0].length;
	}
	
	protected double computeFMeasure(double precision, double recall)
	{
	    if (Utils.eq(precision + recall, 0)) return 0;
	    else return 2 * precision * recall / (precision + recall);
	}


	public abstract double accuracy();
	public abstract double recall();
	public abstract double precision();
	public abstract double fmeasure();
	protected abstract void computeMeasures();

}
