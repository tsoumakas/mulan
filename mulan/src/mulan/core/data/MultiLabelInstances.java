package mulan.core.data;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import mulan.core.MulanRuntimeException;
import mulan.core.WekaException;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

/**
 * Implements multi-label instances data set. Multi-label data are stored in Weka's 
 * {@link Instances}. The class is a convenient wrapper. The data are loaded form  
 * data file, checked for valid format. If hierarchy for labels is specified via 
 * XML meta-data file, the data file is cross-checked with XML for consistency.
 * <br></br>    
 * Applied rules:<br></br>
 * - label names must be unique<br></br>
 * - all labels in XML meta-data must be defined also in ARFF data set<br></br>
 * - each label attribute must be nominal with binary values<br></br>
 * - if labels has hierarchy, then if child labels indicates <code>true</code> of some
 *   data instance, then all its parent labels must indicate also <code>true</code> for that instance<br></br>
 * - the label attributes are moved to the end of data set
 * 
 * @author Jozef Vilcek
 */
public class MultiLabelInstances {
	
	private Instances dataSet;
	private final LabelsMetaDataImpl labelsMetaData;
	
	
	/**
	 * Creates a new instance of {@link MultiLabelInstances} data.
	 * The label attributes are assumed to be at the end of ARFF data file. The count
	 * is specified by parameter. Based on these attributes the {@link LabelsMetaData}
	 * are created.
	 * 
	 * @param arffFilePath the path to ARFF file containing the data
	 * @param numLabelAttributes the number of ARFF data set attributes which are labels.
	 * @throws IllegalArgumentException if arrfFilePath is null
	 * @throws IllegalArgumentException if numLabelAttribures is less than 2
	 * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
 	 * @throws DataLoadException if ARFF data file can not be loaded  
	 */
	public MultiLabelInstances(String arffFilePath, int numLabelAttributes) throws InvalidDataFormatException {
		
		if(arffFilePath == null){
			throw new IllegalArgumentException("The arffFilePath is null.");
		}
		if(numLabelAttributes < 2){
			throw new IllegalArgumentException("The number of label attributes must me at least 2 or higher.");
		}
				
		File arffFile = new File(arffFilePath);
		Instances data = loadInstances(arffFile);
		
		LabelsMetaDataImpl labelsData = new LabelsMetaDataImpl(); 
		int numAttributes = data.numAttributes();
        for (int index = numAttributes - numLabelAttributes; index < numAttributes; index++)
        {
        	String attrName = data.attribute(index).name(); 
        	labelsData.addRootNode(new LabelNodeImpl(attrName));
        }
		
        if(labelsData.getNumLabels() < numLabelAttributes){
        	throw new InvalidDataFormatException("The names of label attributes are not unique.");
        }
		
		validate(data, labelsData);
		
		dataSet = data;
		labelsMetaData = labelsData;
	}
	
	/**
	 * Creates a new instance of {@link MultiLabelInstances} data.
	 * The ARFF data file and labels meta-data are loaded separately. The load failure is
	 * indicated by {@link DataLoadException}. When data are loaded, validations are applied
	 * to ensure consistency between ARFF data and specified labels meta-data.  
	 * 
	 * @param arffFilePath the path to ARFF file containing the data
	 * @param xmlLabelsDefFilePath the path to XML file containing labels meta-data 
	 * @throws IllegalArgumentException if input parameters are null or reference non-existing files
	 * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
	 * @throws DataLoadException if XML meta-data of ARFF data file can not be loaded  
	 */
	public MultiLabelInstances(String arffFilePath, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		
		if(arffFilePath == null){
			throw new IllegalArgumentException("The arffFilePath is null.");
		}
		if(xmlLabelsDefFilePath == null){
			throw new IllegalArgumentException("The xmlLabelsDefFilePath is null.");
		}
		
		File arffFile = new File(arffFilePath);
		Instances data = loadInstances(arffFile);
		LabelsMetaDataImpl labelsData = loadLabesMeta(xmlLabelsDefFilePath);
		
		validate(data, labelsData);
		dataSet = data;
		labelsMetaData = labelsData;
	}
	
	/**
	 * Creates a new instance of {@link MultiLabelInstances} data from existing {@link Instances}
	 * and {@link LabelsMetaData}. The input parameters are not copied. Internally are stored only
	 * references.<br></br>
	 * The data set and labels meta data are validated against each other. Any violation of 
	 * validation criteria result in {@link InvalidDataFormatException}. 
	 * 
	 * @param dataSet the data set with data instances in multi-label format
	 * @param labelsMetaData the meta-data about label attributes of data set
 	 * @throws IllegalArgumentException if input parameters are null
	 * @throws InvalidDataFormatException if multi-label data format is not valid
	 */
	public MultiLabelInstances(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException {
		if(dataSet == null){
			throw new IllegalArgumentException("The dataSet is null.");
		}
		if(labelsMetaData == null){
			throw new IllegalArgumentException("The labelsMetaData is null.");
		}
		
		validate(dataSet, labelsMetaData);
		this.dataSet = dataSet;
		this.labelsMetaData = (LabelsMetaDataImpl)labelsMetaData;
	}
	
	/**
	 * Gets the number of labels (label attributes) 
	 * @return number of labels
	 */
	public int getNumLabels(){
		return labelsMetaData.getNumLabels();
	}

	public Set<Attribute> getLabelAttributes() {
		Set<String> labelNames = labelsMetaData.getLabelNames();
		Set<Attribute> labelAttributes = new HashSet<Attribute>(getNumLabels());
		int numAttributes = dataSet.numAttributes();
		for(int index=0; index < numAttributes; index ++){
			Attribute attr = dataSet.attribute(index);
			if(labelNames.contains(attr.name())){
				labelAttributes.add(attr);
			}
		}
		return labelAttributes;
	}
	
	public Set<Attribute> getFeatureAttributes(){
		Set<String> labelNames = labelsMetaData.getLabelNames();
		Set<Attribute> featureAttributes = new HashSet<Attribute>(getNumLabels());
		int numAttributes = dataSet.numAttributes();
		for(int index=0; index < numAttributes; index ++){
			Attribute attr = dataSet.attribute(index);
			if(!labelNames.contains(attr.name())){
				featureAttributes.add(attr);
			}
		}
		return featureAttributes;
	}

	/**
	 * Gets the {@link LabelsMetaData} instance, which contains descriptive meta-data about
	 * label attributes stored in underlying {@link Instances} data set.
	 *   
	 * @return
	 */
	public LabelsMetaData getLabelsMetaData() {
		return labelsMetaData;
	}

	/**
	 * Gets underlying {@link Instances}, which contains all data.
	 *  
	 * @return
	 */
	public Instances getDataSet(){
		return dataSet;
	}

	/**
	 * If {@link Instances} data set are retrieved from {@link MultiLabelInstances} and 
	 * post-processed, modified by custom code, it can be again reintegrated into 
	 * {@link MultiLabelInstances} if needed. The underlying {@link LabelsMetaData} are 
	 * modified to reflect changes in data set. The method creates new instance of 
	 * {@link MultiLabelInstances} with modified data set and new meta-data.
	 * <br></br>
	 * The supported changes are:<br></br>
	 * - remove of label {@link Attribute} to the existing {@link Instances}<br></br>
	 * - add/remove of {@link Instance} from the existing {@link Instances}<br></br>
	 * - add/remove of feature/predictor {@link Attribute} to the existing {@link Instances}<br></br>
	 *  
	 * @param modifiedDataSet the modified data set
	 * @return 
	 * @throws IllegalArgumentException if specified modified data set is null
	 * @throws InvalidDataFormatException if multi-label data format with specified modifications is not valid 
	 */
	public MultiLabelInstances reintegrateModifiedDataSet(Instances modifiedDataSet) throws InvalidDataFormatException{
		if(modifiedDataSet == null){
			throw new IllegalArgumentException("The modified data set is null.");
		}
		
		//TODO: add support for addition of label attributes to modified data set if necessary
		
		LabelsMetaDataImpl newMetaData = (LabelsMetaDataImpl)labelsMetaData.clone();
		Set<String> origLabelNames = labelsMetaData.getLabelNames();
		for(String labelName : origLabelNames){
			if(modifiedDataSet.attribute(labelName) == null){
				newMetaData.removeLabelNode(labelName);
			}
		}
		
		return new MultiLabelInstances(modifiedDataSet, newMetaData);
	}
	
	/**
	 * Returns a deep copy of the {@link MultiLabelInstances} instance.
	 */
	@Override
	public MultiLabelInstances clone(){
		LabelsMetaData metaDataCopy = labelsMetaData.clone();
		Instances dataSetCopy = new Instances(dataSet);
		try {
			return new MultiLabelInstances(dataSetCopy, metaDataCopy);
		} catch (InvalidDataFormatException ex) {
			throw new MulanRuntimeException(
					String.format("The cloning of '%' class instance failed", getClass()), ex);
		}
	}
	
	private Instances loadInstances(File arffFile){
		if (!arffFile.exists()) {
			throw new IllegalArgumentException(
					String.format("The arff data file does not exists under specified path '%s'.",
							arffFile.getAbsolutePath()));
		}
		
		Instances dataSet = null;
		try {
			FileReader reader = new FileReader(arffFile);
			dataSet = new Instances(reader);
		} catch (IOException exception) {
			throw new DataLoadException(
					String.format("Error loading arff data file '%s'.", arffFile.getAbsolutePath()), exception);
		}
		return dataSet;
	}

	private LabelsMetaDataImpl loadLabesMeta(String xmlLabelsDefFilePath){
		LabelsMetaDataImpl labelsMeta = null;
		try {
			labelsMeta = LabelsBuilder.createLabels(xmlLabelsDefFilePath);
		} catch (LabelsBuilderException exception) {
			throw new DataLoadException(
					String.format("Error loading labels meta-data from xml file '%s'.", xmlLabelsDefFilePath), exception);
		}
		return labelsMeta;
	}
	
	/**
	 * Reorders the labels. All the label attributes are moved to the end of 
	 * data set. This method is introduced only for algorithms which relies on 
	 * label being at the end of data set. Such dependency on internal data structure
	 * should be removed and this method become obsolete.
	 */
	public void reorderLabels() {
		
		// check if label attributes are the last ones in dataSet
		int numAttributes = dataSet.numAttributes();
		Set<String> labelNames = labelsMetaData.getLabelNames();
		boolean mustReorder = false;
		int indexThreshold = numAttributes - labelsMetaData.getNumLabels();
		for(String labelName : labelNames){
			Attribute attribute = dataSet.attribute(labelName);
			if(attribute.index() < indexThreshold){
				mustReorder = true;
				break;
			}
		}
		if(mustReorder){
			LinkedList<Integer> attrOrder = new LinkedList<Integer>();
			for(int index=0;index<numAttributes; index++){
				Attribute attr = dataSet.attribute(index);
				if(labelsMetaData.containsLabel(attr.name())){
					attrOrder.addLast(index);
				}
				else{
					attrOrder.addFirst(index);
				}
			}
			
			int[] orderArray = new int[numAttributes];
			for(int index=0; index < numAttributes; index++)
				orderArray[index] = attrOrder.get(index);
			
			Reorder reorderFilter = new Reorder();
			try{
				reorderFilter.setAttributeIndicesArray(orderArray);
				reorderFilter.setInputFormat(dataSet);
				dataSet = Filter.useFilter(dataSet, reorderFilter);
			}
			catch(Exception ex){
				throw new WekaException("The reorder attrubutes failed when trying to move label attribute at the end of data set.");
			}
		}
	}
	
	/**
	 * Does validation and integrity checks between data set and meta-data. The appropriate exception is
	 * thrown if any inconsistencies of validation rules breached. 
	 * The passed data set and meta-data are not modified in any way.
	 */
	private void validate(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException{
		Set<String> labelNames = labelsMetaData.getLabelNames();
		int numAttributes = dataSet.numAttributes();
		int numMatches = 0;
		for(int index = 0; index < numAttributes; index ++){
			Attribute attribute = dataSet.attribute(index);
			if(labelNames.contains(attribute.name())){
				numMatches++;
				if(!checkLabelAttributeFormat(attribute)){
					throw new InvalidDataFormatException(
							String.format("The format of label attribute '%s' is not valid.", attribute.name()));
				}
			}
		}
		
		if(numMatches != labelNames.size()){
			throw new InvalidDataFormatException(
					String.format("Not all labels defined in meta-data are present in ARFF data file."));
		}
		
		if(labelsMetaData.IsHierarchy()){
			checkLabelsConsistency(dataSet, labelsMetaData.getRootLabels());
		}
	}
	
	// Checks label attribute, if it is nominal and have binary values.
	private boolean checkLabelAttributeFormat(Attribute attribute){
		
		if(attribute.isNominal() != true){
			return false;
		}
		
		List<String> allowedValues = new ArrayList<String>();
		allowedValues.add("0");
		allowedValues.add("1");
		
		int numValues = attribute.numValues();
		if(allowedValues.size() != numValues){
			return false;
		}
		
		for(int index = 0; index < numValues; index++){
			String value = attribute.value(index);
			if(allowedValues.contains(value)){
				allowedValues.remove(value);
			}
		}
		
		if(allowedValues.size() != 0){
			return false;
		}
		
		return true;
	}

	// Checks the consistency of labels if there is a hierarchy between them.
	// If child labels is 'true' for some instance, all its parent labels should be
	// also 'true' for the instance.
	private void checkLabelsConsistency(Instances dataSet, Set<LabelNode> rootLabelNodes) throws InvalidDataFormatException{
		int numInstances = dataSet.numInstances();
		for(int index = 0; index < numInstances; index++){
			Instance instance = dataSet.instance(index);
			for(LabelNode labelNode : rootLabelNodes){
				checkSubtreeConsistency(labelNode, instance, true);
			}
		}
	}
	
	private void checkSubtreeConsistency(LabelNode node, Instance instance, boolean canBeLabelSet) throws InvalidDataFormatException{
		boolean isLabelSet = isLabelSet(instance, node.getName());
		if(isLabelSet == true && canBeLabelSet == false){
			throw new InvalidDataFormatException(String.format("Consistency of labels hierarchy is breached for:\nLabel='%s',\nInstance='%s'", node.getName(), instance.toString()));
		}
		if(node.hasChildren()){
			Set<LabelNode> childNodes = node.getChildren();
			for(LabelNode child : childNodes){
				checkSubtreeConsistency(child, instance, isLabelSet);
			}
		}
	}

	private boolean isLabelSet(Instance instance, String labelName){

		// TODO: dataSet.attribute(labelName) is costly. 
		//       can be improved by maintaining the index.
		Attribute attr = instance.dataset().attribute(labelName);
		double value = instance.value(attr);
		return (value == 1) ? true : false; 
	}
}




