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
 *    MultiLabelInstances.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.*;
import java.util.*;
import mulan.core.ArgumentNullException;
import mulan.core.MulanRuntimeException;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Implements multi-label instances data set. Multi-label data are stored in Weka's
 * {@link Instances}. The class is a convenient wrapper. The data are loaded form data file, checked
 * for valid format. If hierarchy for labels is specified via XML meta-data file, the data file is
 * cross-checked with XML for consistency. <br>
 *  Applied rules:<br>
 *  - label names must be unique<br>
 *  - all labels in XML meta-data must be defined also in ARFF data set<br>
 *  - each label attribute must be nominal with binary values<br>
 *  - if labels has hierarchy, then if child labels indicates <code>true</code> of some data
 * instance, then all its parent labels must indicate also <code>true</code> for that instance<br>
 * 
 * 
 * @author Jozef Vilcek
 */
public class MultiLabelInstances implements Serializable {

    private Instances dataSet;
    private final LabelsMetaData labelsMetaData;
    /**
     * This loader is used when the dataset is loaded incrementally (instance by instance)
     */
    private ArffLoader loader;

    /**
     * Creates a new instance of {@link MultiLabelInstances} data. The label attributes are assumed
     * to be at the end of ARFF data file. The count is specified by parameter. Based on these
     * attributes the {@link LabelsMetaData} are created.
     * 
     * @param arffFilePath the path to ARFF file containing the data
     * @param numLabelAttributes the number of ARFF data set attributes which are labels.
     * @throws ArgumentNullException if arrfFilePath is null
     * @throws IllegalArgumentException if numLabelAttribures is less than 2
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if ARFF data file can not be loaded
     */
    public MultiLabelInstances(String arffFilePath, int numLabelAttributes)
            throws InvalidDataFormatException {

        if (arffFilePath == null) {
            throw new ArgumentNullException("arffFilePath");
        }
        if (numLabelAttributes < 2) {
            throw new IllegalArgumentException(
                    "The number of label attributes must me at least 2 or higher.");
        }

        File arffFile = new File(arffFilePath);
        Instances data = loadInstances(arffFile);
        LabelsMetaData labelsData = loadLabesMeta(data, numLabelAttributes, false);

        validate(data, labelsData);

        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data from the supplied
     * {@link InputStream} data source. The data in the stream are assumed to be in ARFF format. The
     * label attributes in ARFF data are assumed to be the last ones. Based on those attributes the
     * {@link LabelsMetaData} are created.
     * 
     * @param arffDataStream the {@link InputStream} data source to load data in ARFF format
     * @param numLabelAttributes the number of last ARFF data set attributes which are labels.
     * @throws ArgumentNullException if {@link InputStream} data source is null
     * @throws IllegalArgumentException if number of labels attributes is less than 2
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if ARFF data can not be loaded
     */
    public MultiLabelInstances(InputStream arffDataStream, int numLabelAttributes)
            throws InvalidDataFormatException {

        if (arffDataStream == null) {
            throw new ArgumentNullException("arffDataStream");
        }
        if (numLabelAttributes < 2) {
            throw new IllegalArgumentException(
                    "The number of label attributes must me at least 2 or higher.");
        }

        Instances data = loadInstances(arffDataStream);
        LabelsMetaData labelsData = loadLabesMeta(data, numLabelAttributes, false);

        validate(data, labelsData);

        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data from the supplied
     * {@link InputStream} data source. The data in the stream are assumed to be in ARFF format. The
     * label attributes in ARFF data are assumed to be either the last or the first ones. Based on
     * those attributes the {@link LabelsMetaData} are created.
     * 
     * @param arffDataStream the {@link InputStream} data source to load data in ARFF format
     * @param numLabelAttributes the number of last ARFF data set attributes which are labels.
     * @param labelsFirst whether the label attributes are the first or the last ones
     * @throws ArgumentNullException if {@link InputStream} data source is null
     * @throws IllegalArgumentException if number of labels attributes is less than 2
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if ARFF data can not be loaded
     */
    public MultiLabelInstances(InputStream arffDataStream, int numLabelAttributes,
            boolean labelsFirst) throws InvalidDataFormatException {

        if (arffDataStream == null) {
            throw new ArgumentNullException("arffDataStream");
        }
        if (numLabelAttributes < 2) {
            throw new IllegalArgumentException(
                    "The number of label attributes must me at least 2 or higher.");
        }

        Instances data = loadInstances(arffDataStream);
        LabelsMetaData labelsData = loadLabesMeta(data, numLabelAttributes, labelsFirst);

        validate(data, labelsData);

        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data. The Instances object and labels
     * meta-data are loaded separately. The load failure is indicated by {@link DataLoadException}.
     * When data are loaded, validations are applied to ensure consistency between ARFF data and
     * specified labels meta-data.
     * 
     * @param data the Instances object containing the data
     * @param xmlLabelsDefFilePath the path to XML file containing labels meta-data
     * @throws IllegalArgumentException if input parameters refers to non-existing files
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if XML meta-data of ARFF data file can not be loaded
     */
    public MultiLabelInstances(Instances data, String xmlLabelsDefFilePath)
            throws InvalidDataFormatException {
        if (xmlLabelsDefFilePath == null) {
            throw new ArgumentNullException("xmlLabelsDefFilePath");
        }
        LabelsMetaData labelsData = loadLabelsMeta(xmlLabelsDefFilePath);
        validate(data, labelsData);
        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data. The ARFF data file and labels
     * meta-data are loaded separately. The load failure is indicated by {@link DataLoadException}.
     * When data are loaded, validations are applied to ensure consistency between ARFF data and
     * specified labels meta-data.
     * 
     * @param arffFilePath the path to ARFF file containing the data
     * @param xmlLabelsDefFilePath the path to XML file containing labels meta-data
     * @throws ArgumentNullException if input parameters are null
     * @throws IllegalArgumentException if input parameters refers to non-existing files
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if XML meta-data of ARFF data file can not be loaded
     */
    public MultiLabelInstances(String arffFilePath, String xmlLabelsDefFilePath)
            throws InvalidDataFormatException {
        if (arffFilePath == null) {
            throw new ArgumentNullException("arffFilePath");
        }
        if (xmlLabelsDefFilePath == null) {
            throw new ArgumentNullException("xmlLabelsDefFilePath");
        }
        File arffFile = new File(arffFilePath);
        Instances data = loadInstances(arffFile);
        LabelsMetaData labelsData = loadLabelsMeta(xmlLabelsDefFilePath);
        validate(data, labelsData);
        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data. The ARFF data file and labels
     * meta-data are loaded separately. The load failure is indicated by {@link DataLoadException}.
     * When data are loaded, validations are applied to ensure consistency between ARFF data and
     * specified labels meta-data.
     * 
     * @param arffFilePath the path to ARFF file containing the data
     * @param xmlLabelsDefFilePath the path to XML file containing labels meta-data
     * @param incremental if incremental or not
     * @throws ArgumentNullException if input parameters are null
     * @throws IllegalArgumentException if input parameters refers to non-existing files
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws IOException if arff or xml is not found
     * @throws DataLoadException if XML meta-data of ARFF data file can not be loaded
     */
    public MultiLabelInstances(String arffFilePath, String xmlLabelsDefFilePath, boolean incremental)
            throws InvalidDataFormatException, IOException {
        if (arffFilePath == null) {
            throw new ArgumentNullException("arffFilePath");
        }
        if (xmlLabelsDefFilePath == null) {
            throw new ArgumentNullException("xmlLabelsDefFilePath");
        }
        // the loader is initialized
        loader = new ArffLoader();
        loader.setFile(new File(arffFilePath));
        // only the structure of the dataset is actually loaded
        Instances data = loader.getStructure();
        LabelsMetaData labelsData = loadLabelsMeta(xmlLabelsDefFilePath);
        validate(data, labelsData);
        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data from the supplied
     * {@link InputStream} data source. The data in the stream are assumed to be in ARFF format. The
     * labels meta data for ARFF data are retrieved separately from the different
     * {@link InputStream} data source. The meta data are assumed to be in XML format and conform to
     * valid schema. Data load load failure is indicated by {@link DataLoadException}. When data are
     * loaded, validations are applied to ensure consistency between ARFF data and specified labels
     * meta-data.
     * 
     * @param arffDataStream the {@link InputStream} data source to load data in ARFF format
     * @param xmlLabelsDefStream the {@link InputStream} data source to load XML labels meta data
     * @throws ArgumentNullException if input parameters are null
     * @throws IllegalArgumentException if input parameters refers to non-existing files
     * @throws InvalidDataFormatException if format of loaded multi-label data is invalid
     * @throws DataLoadException if XML meta-data of ARFF data can not be loaded
     */
    public MultiLabelInstances(InputStream arffDataStream, InputStream xmlLabelsDefStream)
            throws InvalidDataFormatException {

        if (arffDataStream == null) {
            throw new ArgumentNullException("arffDataStream");
        }
        if (xmlLabelsDefStream == null) {
            throw new ArgumentNullException("xmlLabelsDefStream");
        }

        Instances data = loadInstances(arffDataStream);
        LabelsMetaData labelsData = loadLabelsMeta(xmlLabelsDefStream);

        validate(data, labelsData);
        dataSet = data;
        labelsMetaData = labelsData;
    }

    /**
     * Creates a new instance of {@link MultiLabelInstances} data from existing {@link Instances}
     * and {@link LabelsMetaData}. The input parameters are not copied. Internally are stored only
     * references.<br>
     *  The data set and labels meta data are validated against each other. Any violation of
     * validation criteria result in {@link InvalidDataFormatException}.
     * 
     * @param dataSet the data set with data instances in multi-label format
     * @param labelsMetaData the meta-data about label attributes of data set
     * @throws IllegalArgumentException if input parameters are null
     * @throws InvalidDataFormatException if multi-label data format is not valid
     */
    public MultiLabelInstances(Instances dataSet, LabelsMetaData labelsMetaData)
            throws InvalidDataFormatException {
        if (dataSet == null) {
            throw new ArgumentNullException("dataSet");
        }
        if (labelsMetaData == null) {
            throw new ArgumentNullException("labelsMetaData");
        }

        validate(dataSet, labelsMetaData);
        this.dataSet = dataSet;
        this.labelsMetaData = labelsMetaData;
    }

    /**
     * Gets the number of labels (label attributes)
     * 
     * @return number of labels
     */
    public int getNumLabels() {
        return labelsMetaData.getNumLabels();
    }

    /**
     * Gets the number of instances
     * 
     * @return number of instances
     */
    public int getNumInstances() {
        return dataSet.numInstances();
    }

    /**
     * Gets the cardinality of the dataset
     * 
     * @return dataset cardinality
     */
    public double getCardinality() {
        double labelCardinality = 0;

        int numInstances = dataSet.numInstances();
        int numLabels = labelsMetaData.getNumLabels();
        int[] labelIndices = getLabelIndices();

        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numLabels; j++) {
                if (dataSet.instance(i).stringValue(labelIndices[j]).equals("1")) {
                    labelCardinality++;
                }
            }
        }

        labelCardinality /= numInstances;
        return labelCardinality;
    }

    /**
     * @return an array with the indices of the label attributes inside the Instances object
     */
    public int[] getLabelIndices() {
        int[] labelIndices = new int[labelsMetaData.getNumLabels()];
        int numAttributes = dataSet.numAttributes();
        Set<String> labelNames = labelsMetaData.getLabelNames();
        int counter = 0;

        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                labelIndices[counter] = index;
                counter++;
            }
        }

        return labelIndices;
    }

    /**
     * @return an array with the names of the label attributes inside the Instances object
     */
    public String[] getLabelNames() {
        String[] orderedLabelNames = new String[labelsMetaData.getNumLabels()];
        int numAttributes = dataSet.numAttributes();
        Set<String> labelNames = labelsMetaData.getLabelNames();
        int counter = 0;

        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                orderedLabelNames[counter] = attr.name();
                counter++;
            }
        }

        return orderedLabelNames;
    }

    /**
     * @return a mapping of attribute names and their indices Instances object
     */
    public Map<String, Integer> getLabelsOrder() {
        int numAttributes = dataSet.numAttributes();
        Set<String> labelNames = labelsMetaData.getLabelNames();
        HashMap<String, Integer> assoc = new HashMap<String, Integer>();

        int counter = 0;
        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                assoc.put(attr.name(), counter);
                counter++;
            }
        }

        return assoc;
    }

    /**
     * Gets the {@link Set} of label {@link Attribute} instances of this {@link MultiLabelInstances}
     * instance.
     * 
     * @return the Set of label Attribute instances
     */
    public Set<Attribute> getLabelAttributes() {
        Set<String> labelNames = labelsMetaData.getLabelNames();
        Set<Attribute> labelAttributes = new HashSet<Attribute>(getNumLabels());
        int numAttributes = dataSet.numAttributes();
        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                labelAttributes.add(attr);
            }
        }
        return labelAttributes;
    }

    /**
     * Gets the array with indices of feature attributes stored in underlying {@link Instances} data
     * set.
     * 
     * @return an array with the indices of the feature attributes
     */
    public int[] getFeatureIndices() {

        int numAttributes = dataSet.numAttributes();
        Set<Attribute> featureAttributes = getFeatureAttributes();
        int[] featureIndices = new int[featureAttributes.size()];
        int counter = 0;
        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (featureAttributes.contains(attr)) {
                featureIndices[counter] = attr.index();
                counter++;
            }
        }

        return featureIndices;
    }

    /**
     * Gets the {@link Set} of feature {@link Attribute} instances of this
     * {@link MultiLabelInstances} instance.
     * 
     * @return the {@link Set} of feature {@link Attribute} instances
     */
    public Set<Attribute> getFeatureAttributes() {
        Set<String> labelNames = labelsMetaData.getLabelNames();
        Set<Attribute> featureAttributes = new HashSet<Attribute>(getNumLabels());
        int numAttributes = dataSet.numAttributes();
        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (!labelNames.contains(attr.name())) {
                featureAttributes.add(attr);
            }
        }
        return featureAttributes;
    }

    /**
     * Gets the {@link LabelsMetaData} instance, which contains descriptive meta-data about label
     * attributes stored in underlying {@link Instances} data set.
     * 
     * @return descriptive meta-data about label attributes
     */
    public LabelsMetaData getLabelsMetaData() {
        return labelsMetaData;
    }

    /**
     * Gets underlying {@link Instances}, which contains all data.
     * 
     * @return underlying Instances object which contains all data
     */
    public Instances getDataSet() {
        return dataSet;
    }

    /**
     * If {@link Instances} data set are retrieved from {@link MultiLabelInstances} and
     * post-processed, modified by custom code, it can be again reintegrated into
     * {@link MultiLabelInstances} if needed. The underlying {@link LabelsMetaData} are modified to
     * reflect changes in data set. The method creates new instance of {@link MultiLabelInstances}
     * with modified data set and new meta-data. <br>
     *  The supported changes are:<br>
     *  - remove of label {@link Attribute} to the existing {@link Instances}<br>
     *  - add/remove of {@link Instance} from the existing {@link Instances}<br>
     *  - add/remove of feature/predictor {@link Attribute} to the existing {@link Instances}<br>
     * 
     * 
     * @param modifiedDataSet the modified data set
     * @return the modified data set
     * @throws IllegalArgumentException if specified modified data set is null
     * @throws InvalidDataFormatException if multi-label data format with specified modifications is
     *             not valid
     */
    public MultiLabelInstances reintegrateModifiedDataSet(Instances modifiedDataSet)
            throws InvalidDataFormatException {
        if (modifiedDataSet == null) {
            throw new IllegalArgumentException("The modified data set is null.");
        }

        // TODO: add support for addition of label attributes to modified data set if necessary

        LabelsMetaDataImpl newMetaData = (LabelsMetaDataImpl) labelsMetaData.clone();
        Set<String> origLabelNames = labelsMetaData.getLabelNames();
        for (String labelName : origLabelNames) {
            if (modifiedDataSet.attribute(labelName) == null) {
                newMetaData.removeLabelNode(labelName);
            }
        }

        return new MultiLabelInstances(modifiedDataSet, newMetaData);
    }

    /**
     * Returns a deep copy of the {@link MultiLabelInstances} instance.
     */
    @Override
    public MultiLabelInstances clone() {
        LabelsMetaData metaDataCopy = labelsMetaData.clone();
        Instances dataSetCopy = new Instances(dataSet);
        try {
            return new MultiLabelInstances(dataSetCopy, metaDataCopy);
        } catch (InvalidDataFormatException ex) {
            throw new MulanRuntimeException(String.format(
                    "The cloning of '%' class instance failed", getClass()), ex);
        }
    }

    private Instances loadInstances(File arffFile) {
        if (!arffFile.exists()) {
            throw new IllegalArgumentException(String.format(
                    "The arff data file does not exists under specified path '%s'.",
                    arffFile.getAbsolutePath()));
        }

        Instances aDataSet = null;
        FileInputStream fileStream = null;
        try {
            fileStream = new FileInputStream(arffFile);
        } catch (FileNotFoundException exception) {
            throw new DataLoadException(String.format(
                    "The specified data file '%s' can not be found.", arffFile.getAbsolutePath()),
                    exception);
        }

        aDataSet = loadInstances(fileStream);

        return aDataSet;
    }

    private Instances loadInstances(InputStream stream) {
        Instances aDataSet = null;
        InputStreamReader streamReader = new InputStreamReader(stream);
        try {
            aDataSet = new Instances(streamReader);
        } catch (IOException exception) {
            throw new DataLoadException(String.format(
                    "Error creating Instances data from supplied Reader data source: "
                            + exception.getMessage(), exception));
        }
        return aDataSet;
    }

    private LabelsMetaData loadLabelsMeta(String xmlLabelsDefFilePath) {
        LabelsMetaData labelsMeta = null;
        try {
            labelsMeta = LabelsBuilder.createLabels(xmlLabelsDefFilePath);
        } catch (LabelsBuilderException exception) {
            throw new DataLoadException(String.format(
                    "Error loading labels meta-data from xml file '%s'.", xmlLabelsDefFilePath),
                    exception);
        }
        return labelsMeta;
    }

    private LabelsMetaData loadLabelsMeta(InputStream xmlLabelsDefStream) {
        LabelsMetaData labelsMeta = null;
        try {
            labelsMeta = LabelsBuilder.createLabels(xmlLabelsDefStream);
        } catch (LabelsBuilderException exception) {
            throw new DataLoadException(
                    String.format("Error loading labels meta-data from input stream."), exception);
        }
        return labelsMeta;
    }

    private LabelsMetaData loadLabesMeta(Instances data, int numLabels, boolean labelsFirst)
            throws InvalidDataFormatException {
        LabelsMetaDataImpl labelsData = new LabelsMetaDataImpl();
        int numAttributes = data.numAttributes();
        if (labelsFirst) {
            for (int index = 0; index < numLabels; index++) {
                String attrName = data.attribute(index).name();
                labelsData.addRootNode(new LabelNodeImpl(attrName));
            }
        } else {
            for (int index = numAttributes - numLabels; index < numAttributes; index++) {
                String attrName = data.attribute(index).name();
                labelsData.addRootNode(new LabelNodeImpl(attrName));
            }
        }

        if (labelsData.getNumLabels() < numLabels) {
            throw new InvalidDataFormatException("The names of label attributes are not unique.");
        }

        return labelsData;
    }

    /**
     * Does validation and integrity checks between data set and meta-data. The appropriate
     * exception is thrown if any inconsistencies of validation rules breached. The passed data set
     * and meta-data are not modified in any way.
     */
    private void validate(Instances dataSet, LabelsMetaData labelsMetaData)
            throws InvalidDataFormatException {
        Set<String> labelNames = labelsMetaData.getLabelNames();
        if (labelNames.size() < 2) {
            throw new InvalidDataFormatException(
                    String.format(
                            "There must be at least 2 label attributes specified, but only '%s' are defined in metadata",
                            labelNames.size()));
        }

        int numAttributes = dataSet.numAttributes();
        Set<String> labelNamesInArff = new HashSet<String>();
        for (int index = 0; index < numAttributes; index++) {
            Attribute attribute = dataSet.attribute(index);
            if (labelNames.contains(attribute.name())) {
                labelNamesInArff.add(attribute.name());
                if (!checkLabelAttributeFormat(attribute)) {
                    throw new InvalidDataFormatException(String.format(
                            "The format of label attribute '%s' is not valid.", attribute.name()));
                }
            }
        }

        if (labelNamesInArff.size() < labelNames.size()) {
            String lbabelNamesNotInArff = "";
            for (String labelName : labelNames) {
                if (!labelNamesInArff.contains(labelName)) {
                    lbabelNamesNotInArff += "\'" + labelName + "\' ";
                }
            }
            throw new InvalidDataFormatException(
                    String.format("Not all labels defined in meta-data are present in ARFF data file. Label(s) missing: "
                            + lbabelNamesNotInArff));
        }

        if (labelsMetaData.isHierarchy()) {
            checkLabelsConsistency(dataSet, labelsMetaData.getRootLabels());
        }
    }

    // Checks label attribute, if it is numeric or nominal and has binary values.
    private boolean checkLabelAttributeFormat(Attribute attribute) {

        if (attribute.isNumeric() == true) {
            return true;
        }

        if (attribute.isNominal() != true) {
            return false;
        }

        List<String> allowedValues = new ArrayList<String>();
        allowedValues.add("0");
        allowedValues.add("1");

        int numValues = attribute.numValues();
        if (allowedValues.size() != numValues) {
            return false;
        }

        for (int index = 0; index < numValues; index++) {
            String value = attribute.value(index);
            if (allowedValues.contains(value)) {
                allowedValues.remove(value);
            }
        }

        if (!allowedValues.isEmpty()) {
            return false;
        }

        return true;
    }

    // Checks the consistency of labels if there is a hierarchy between them.
    // If child labels is 'true' for some instance, all its parent labels should
    // be also 'true' for the instance.
    private void checkLabelsConsistency(Instances dataSet, Set<LabelNode> rootLabelNodes)
            throws InvalidDataFormatException {
        // create an index for faster access to attribute based on name
        Map<String, Attribute> attributesIndex = new HashMap<String, Attribute>();
        for (int index = 0; index < dataSet.numAttributes(); index++) {
            Attribute attribute = dataSet.attribute(index);
            attributesIndex.put(attribute.name(), attribute);
        }

        int numInstances = dataSet.numInstances();
        for (int index = 0; index < numInstances; index++) {
            Instance instance = dataSet.instance(index);
            for (LabelNode labelNode : rootLabelNodes) {
                checkSubtreeConsistency(labelNode, instance, true, attributesIndex);
            }
        }
    }

    private void checkSubtreeConsistency(LabelNode node, Instance instance, boolean canBeLabelSet,
            Map<String, Attribute> attributesIndex) throws InvalidDataFormatException {
        boolean isLabelSet = isLabelSet(instance, node.getName(), attributesIndex);
        if (isLabelSet == true && canBeLabelSet == false) {
            throw new InvalidDataFormatException(String.format(
                    "Consistency of labels hierarchy is breached for: Label='%s', Instance='%s'",
                    node.getName(), instance.toString()));
        }
        if (node.hasChildren()) {
            Set<LabelNode> childNodes = node.getChildren();
            for (LabelNode child : childNodes) {
                checkSubtreeConsistency(child, instance, isLabelSet, attributesIndex);
            }
        }
    }

    private boolean isLabelSet(Instance instance, String labelName,
            Map<String, Attribute> attributesIndex) {
        if (instance.stringValue(attributesIndex.get(labelName)).equals("1"))
            return true;
        else
            return false;
    }

    /**
     * Create a HashMap that contains every label, with its depth in the Hierarchical tree
     * 
     * @return a HashMap that contains every label with its depth in the Hierarchical tree
     */
    public HashMap<String, Integer> getLabelDepth() {
        int numAttributes = dataSet.numAttributes();
        Set<String> labelNames = labelsMetaData.getLabelNames();
        HashMap<String, Integer> assoc = new HashMap<String, Integer>();

        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                assoc.put(attr.name(), getDepth(attr.name()));
            }
        }
        return assoc;
    }

    /**
     * Calculates the depth of a label, in the Hierarchy of the tree of labels. Returns the counter
     * of every level. We define the root node label that has the depth 1
     * 
     * @param labelName the name of the label
     * @return the depth of a label
     */
    public int getDepth(String labelName) {
        int counter = 0;

        while (labelsMetaData.getLabelNode(labelName).hasParent()) {
            counter++;
            labelName = labelsMetaData.getLabelNode(labelName).getParent().getName();
        }
        return counter + 1;
    }

    /**
     * Returns the depth of the labels
     * 
     * @return the depth of the labels
     */
    public int[] getLabelDepthIndices() {
        int[] labelDepthIndices = new int[labelsMetaData.getNumLabels()];
        int numAttributes = dataSet.numAttributes();
        Set<String> labelNames = labelsMetaData.getLabelNames();
        int counter = 0;

        for (int index = 0; index < numAttributes; index++) {
            Attribute attr = dataSet.attribute(index);
            if (labelNames.contains(attr.name())) {
                labelDepthIndices[counter] = getDepth(attr.name());
                counter++;
            }
        }

        return labelDepthIndices;
    }

    /**
     * Method that checks whether an instance has missing labels
     * 
     * @param instance one instance of this dataset
     * @return true if the instance has missing labels
     */
    public boolean hasMissingLabels(Instance instance) {
        int numLabels = getNumLabels();
        int[] labelIndices = getLabelIndices();

        boolean missing = false;
        for (int j = 0; j < numLabels; j++) {
            if (instance.isMissing(labelIndices[j])) {
                missing = true;
                break;
            }
        }
        return missing;
    }

    /**
     * Returns the next instace of a multi-label dataset when the incremental read is enabled.
     * 
     * @return The next instace of a multi-label dataset
     * @throws IOException if loader fails
     */
    public Instance getNextInstance() throws IOException {
        if (loader == null) {
            throw new DataLoadException(
                    "Dataset was not loaded incrementally. Use the incremental constructor for this purpose.");
        }
        return loader.getNextInstance(dataSet);
    }
}
