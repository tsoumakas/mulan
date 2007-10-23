package mulan.evaluation;

/**
 * CrossValidation - has identical semantics with Evaluation.
 * User is passed an instance of this class when calling
 * Evaluator.crossValidate() and friends.
 */
public class CrossValidation extends Evaluation
{

	protected int numFolds;
	
	//TODO: add some stratification options?
	protected CrossValidation(LabelBasedEvaluation labelBased,
			ExampleBasedEvaluation exampleBased, int numFolds)
	{
		super(labelBased, exampleBased);
		this.numFolds = numFolds;
	}

	public int numFolds()
	{
		return numFolds;
	}
}
