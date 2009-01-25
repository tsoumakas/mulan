package mulan.evaluation;

import java.util.Collections;
import java.util.List;

/**
 * Used to store data from multi-label learner model examination for evaluation purposes.
 * The class provides output of the model, which is the model response to presented 
 * input data instance and a true labels for instance the model was examined with.
 * 
 * @author Jozef Vilcek
 */
class ModelEvaluationDataPair<T> {
	
	private final T modelOutput;
	private List<Boolean> trueLabels;
	
	/**
	 * Creates a new instance. 
	 *  
	 * @param modelOutput model output
	 * @param trueLabels the true labels bipartition
	 */
	ModelEvaluationDataPair(T modelOutput, List<Boolean> trueLabels){
		if(modelOutput == null || trueLabels == null) {
			throw new IllegalArgumentException("Neither modelOutput or trueLabels can be null.");
		}
		
		if(trueLabels.size() <= 1){
			throw new IllegalArgumentException("trueLabels is too less in dimension.");
		}
		
		this.modelOutput = modelOutput;
		this.trueLabels = trueLabels;
	}
						    
	
	/**
	 * Returns read-only list of model output.
	 * @return model output
	 */
	T getModelOutput(){
		return modelOutput;
	}
	
	/**
	 * Returns read-only list of true labels the model should output.
	 * @return true labels
	 */
	List<Boolean> getTrueLabels(){
		return Collections.unmodifiableList(trueLabels);
	}
	

}
