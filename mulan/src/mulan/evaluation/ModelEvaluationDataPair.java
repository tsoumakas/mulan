package mulan.evaluation;

import java.util.Collections;
import java.util.List;

/**
 * Used to store data from multi-label learner model examination for evaluation purposes.
 * The class provides output of the model, which is the model response to presented 
 * input data instance, an actual output, which is a true (ideal) output the model should 
 * provide when presented with particular input data and confidences for each output item 
 * of the model result.
 * 
 * @author Jozef Vilcek
 */
class ModelEvaluationDataPair<T> {
	
	private List<T> modelOutput;
	private List<Double> modelConfidences;
	private List<T> actualOutput;
	
	
	/**
	 * Creates a new instance. 
	 *  
	 * @param modelOutput model output
	 * @param actualOutput actual output the model should output
	 */
	ModelEvaluationDataPair(List<T> modelOutput, List<T> actualOutput){
		if(modelOutput == null || actualOutput == null) {
			throw new IllegalArgumentException("Neither modelOutput or actualOutput can be null.");
		}
		int modelOutputSize = modelOutput.size();
		int actualOutputSize = actualOutput.size();
		if(modelOutputSize == 0 || actualOutputSize == 0){
			throw new IllegalArgumentException("Either modelOutput or actualOutput does not contain any " +
							"items. Empty collections are not allowed.");
		}
		if(modelOutputSize != actualOutputSize){
			throw new IllegalArgumentException("Both modelOutput and actualOutput must have same dimenstions.");
		}
		this.modelOutput = modelOutput;
		this.actualOutput = actualOutput;
	}
						    
	
	/**
	 * Creates a new instance. 
	 *  
	 * @param modelOutput model output
	 * @param actualOutput actual output the model should output
	 * @param modelConfidences confidences for model output
	 */
	ModelEvaluationDataPair(List<T> modelOutput, List<T> actualOutput, 
						      List<Double> modelConfidences) {
		this(modelOutput, actualOutput);
		
		if(modelConfidences == null){
			throw new IllegalArgumentException("Input parameter modelConfidences is null.");
		}
		if(modelConfidences.size() != modelOutput.size()){
			throw new IllegalArgumentException("The dimension of modelConfidences does not match model output.");
		}
		this.modelConfidences = modelConfidences;
	}
	
	/**
	 * Returns read-only list of model output.
	 * @return model output
	 */
	List<T> getModelOutput(){
		return Collections.unmodifiableList(modelOutput);
	}
	
	/**
	 * Returns read-only list of confidences of model output.
	 * @return confidences of model output or null if they are not specified
	 */
	List<Double> getModelConfidences(){
		return (modelConfidences == null) ? 
				null : Collections.unmodifiableList(modelConfidences);
	}
	
	/**
	 * Returns read-only list of actual output the model should output.
	 * @return actual output
	 */
	List<T> getActualOutput(){
		return Collections.unmodifiableList(actualOutput);
	}
	

}
