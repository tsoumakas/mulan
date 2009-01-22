package mulan.evaluation;

import java.util.Dictionary;
import java.util.Hashtable;
import java.util.List;

/**
 * Used to store data sets from cross validation of multi-label learner model. Results of particular 
 * cross validation fold are represented as list of {@link ModelEvaluationDataPair} items. This class
 * represents just a grouping of all cross validations of the same multi-label learner, so they can be 
 * reused as whole by other components in evaluation.
 *   
 * @author Jozef Vilcek
 */
class ModelCrossValidationDataSet<T> {

	/**
	 * Set of items, where one item is a list data pairs obtained as a 
	 * response of multi-label learner in one cross validation fold
	 */
	private Dictionary<Integer, List<ModelEvaluationDataPair<T>>> crossValidationData;
	
	/**
	 * Creates a new instance.
	 */
	public ModelCrossValidationDataSet() {
		crossValidationData = new Hashtable<Integer, List<ModelEvaluationDataPair<T>>>();
	}
	
	/**
	 * Adds a data pairs (results) obtained from the multi-label 
	 * learner in particular cross validation fold.
	 * If any data pairs are already set for specified fold number, 
	 * they will be replaced by new data pairs. 
	 * 
	 * @param foldNumber the number of fold in which data were obtained
	 * @param dataPairs the data pairs obtained during multi-label learner examination
	 */
	void addFoldData(int foldNumber, List<ModelEvaluationDataPair<T>> dataPairs){
		crossValidationData.put(foldNumber, dataPairs);
	}
	
	/**
	 * Removes specified fold number and data associated with it from this data set.
	 * If specified fold is not in data set, the method does nothing. 
	 * 
	 * @param foldNumber the number of fold to be removed
	 */
	void removeFoldData(int foldNumber){
		crossValidationData.remove(foldNumber);
	}
	
	/**
	 * Returns data pairs of specified cross validation fold number.
	 * 
	 * @param foldNumber the fold number
	 * @return the data pairs of specified cross validation fold
	 */
	List<ModelEvaluationDataPair<T>> getFoldData(int foldNumber){
		return crossValidationData.get(foldNumber);
	}
	
	/**
	 * Returns the number of cross validation folds present in data set.
	 * 
	 * @return the number of folds
	 */
	int getNumFolds(){
		return crossValidationData.size();
	}
}
