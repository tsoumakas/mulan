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
package mulan.classifier.meta;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Source;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import mulan.data.InvalidDataFormatException;
import mulan.data.LabelNode;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaData;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.data.DataUtils;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Class that builds a hierarchy on flat lables of given mulltilabel data.
 * The hierarchy may be built with three methods.
 *
 * @author George Saridis
 * @author Grigorios Tsoumakas
 * @version 0.1
 */
public class HierarchyBuilder implements Serializable {

    private int numPartitions;
    private Document labelsXMLDoc;
    private Method method;

    /**
     * Constructs a new istance based on given number of partitions and method
     * 
     * @param partitions the number of partitions
     * @param method the partitioning method
     */
    public HierarchyBuilder(int partitions, Method method) {
        numPartitions = partitions;
        this.method = method;
    }

    /**
     * Builds a hierarhical multi-label dataset. Firstly a random hierarchy is
     * built on top of the labels of a flat multi-label dataset, by recursively
     * randomly partitioning the labels into a specified number of clusters.
     * Then the values for the new "meta-labels" are properly set, so that
     * the hierarchy is respected.
     *
     * @param mlData the multiLabel data on which the new hierarchy will be built
     * @return the new multiLabel data
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelInstances buildHierarchy(MultiLabelInstances mlData) throws Exception {
        LabelsMetaData labelsMetaData = buildLabelHierarchy(mlData);
        return HierarchyBuilder.createHierarchicalDataset(mlData, labelsMetaData);
    }

    /**
     * Builds a hierarhy of labels on top of the labels of a flat multi-label
     * dataset, by recursively partitioning the labels into a specified number
     * of partitions.
     *
     * @param mlData the multiLabel data on with the new hierarchy will be built
     * @return a hierarchy of labels
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public LabelsMetaData buildLabelHierarchy(MultiLabelInstances mlData) throws Exception {
        if (numPartitions > mlData.getNumLabels()) {
            throw new IllegalArgumentException("Number of labels is smaller than the number of partitions");
        }

        Set<String> setOfLabels = mlData.getLabelsMetaData().getLabelNames();
        List<String> listOfLabels = new ArrayList<String>();
        for (String label : setOfLabels) {
            listOfLabels.add(label);
        }

        ArrayList<String>[] childrenLabels = null;
        switch (method) {
            case Random:
                childrenLabels = randomPartitioning(numPartitions, listOfLabels);
                break;
            case Clustering:
                childrenLabels = clustering(numPartitions, listOfLabels, mlData, false);
                break;
            case BalancedClustering:
                childrenLabels = clustering(numPartitions, listOfLabels, mlData, true);
                break;
        }

        for (int i = 0; i < numPartitions; i++) {
            if (childrenLabels[i].size() == listOfLabels.size()) {
                // another idea is to add leaves here
                childrenLabels = randomPartitioning(numPartitions, listOfLabels);
                break;
            }
        }

        LabelsMetaDataImpl metaData = new LabelsMetaDataImpl();
        for (int i = 0; i < numPartitions; i++) {
            if (childrenLabels[i].isEmpty()) {
                continue;
            }
            if (childrenLabels[i].size() == 1) {
                metaData.addRootNode(new LabelNodeImpl(childrenLabels[i].get(0)));
                continue;
            }
            if (childrenLabels[i].size() > 1) {
                LabelNodeImpl metaLabel = new LabelNodeImpl("MetaLabel " + (i + 1));
                createLabelsMetaDataRecursive(metaLabel, childrenLabels[i], mlData);
                metaData.addRootNode(metaLabel);
            }
        }

        return metaData;
    }

    /**
     * Builds the hierarchy and constructs auxiliary files
     * 
     * @param mlData the flat training data
     * @param arffName the name of the hierachical data
     * @param xmlName the filename for the hirearchy
     * @return the hierarchical data
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelInstances buildHierarchyAndSaveFiles(MultiLabelInstances mlData, String arffName, String xmlName) throws Exception {
        MultiLabelInstances newData = buildHierarchy(mlData);
        saveToArffFile(newData.getDataSet(), new File(arffName));
        createXMLFile(mlData.getLabelsMetaData());
        saveToXMLFile(xmlName);
        return newData;
    }

    private void createLabelsMetaDataRecursive(LabelNodeImpl node, List<String> labels, MultiLabelInstances mlData) {
        if (labels.size() <= numPartitions) {
            for (int i = 0; i < labels.size(); i++) {
                LabelNodeImpl child = new LabelNodeImpl(labels.get(i));
                node.addChildNode(child);
            }
            return;
        }

        ArrayList<String>[] childrenLabels = null;
        switch (method) {
            case Random:
                childrenLabels = randomPartitioning(numPartitions, labels);
                break;
            case Clustering:
                childrenLabels = clustering(numPartitions, labels, mlData, false);
                break;
            case BalancedClustering:
                childrenLabels = clustering(numPartitions, labels, mlData, true);
                break;
        }

        for (int i = 0; i < numPartitions; i++) {
            if (childrenLabels[i].size() == labels.size()) {
                // another idea is to add leaves here
                childrenLabels = randomPartitioning(numPartitions, labels);
                break;
            }
        }

        for (int i = 0; i < numPartitions; i++) {
            if (childrenLabels[i].isEmpty()) {
                continue;
            }
            if (childrenLabels[i].size() == 1) {
                LabelNodeImpl child = new LabelNodeImpl(childrenLabels[i].get(0));
                node.addChildNode(child);
                continue;
            }
            if (childrenLabels[i].size() > 1) {
                LabelNodeImpl child = new LabelNodeImpl(node.getName() + "." + (i + 1));
                node.addChildNode(child);
                createLabelsMetaDataRecursive(child, childrenLabels[i], mlData);
            }
        }
    }

    private ArrayList<String>[] clustering(int clusters, List<String> labels, MultiLabelInstances mlData, boolean balanced) {
        ArrayList<String>[] childrenLabels = new ArrayList[clusters];
        for (int i = 0; i < clusters; i++) {
            childrenLabels[i] = new ArrayList<String>();
        }

        // transpose data and keep only labels in the parameter list
        int numInstances = mlData.getDataSet().numInstances();
        ArrayList<Attribute> attInfo = new ArrayList<Attribute>(numInstances);
        for (int i = 0; i < numInstances; i++) {
            Attribute att = new Attribute("instance" + (i + 1));
            attInfo.add(att);
        }
        System.out.println(new Date() + " constructing instances");
        Instances transposed = new Instances("transposed", attInfo, 0);
        int[] labelIndices = mlData.getLabelIndices();
        for (int i = 0; i < labels.size(); i++) {
            int index = -1;
            for (int k=0; k<labelIndices.length; k++) {
                if (mlData.getDataSet().attribute(labelIndices[k]).name().equals(labels.get(i))) {
                    index = labelIndices[k];
                }
            }
            double[] values = new double[numInstances];
            for (int j = 0; j < numInstances; j++) {
                values[j] = mlData.getDataSet().instance(j).value(index);
            }
            Instance newInstance = DataUtils.createInstance(mlData.getDataSet().instance(0), 1, values);
            transposed.add(newInstance);
        }

        if (!balanced) {
            EM clusterer = new EM();
            try {
                // cluster the labels
                clusterer.setNumClusters(clusters);
                System.out.println("clustering");
                clusterer.buildClusterer(transposed);
                // return the clustering
                for (int i = 0; i < labels.size(); i++) {
                    childrenLabels[clusterer.clusterInstance(transposed.instance(i))].add(labels.get(i));
                }
            } catch (Exception ex) {
                Logger.getLogger(HierarchyBuilder.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            ConstrainedKMeans clusterer = new ConstrainedKMeans();
            try {
                // cluster the labels
                clusterer.setMaxIterations(20);
                clusterer.setNumClusters(clusters);
                System.out.println("balanced clustering");
                clusterer.buildClusterer(transposed);
                // return the clustering
                for (int i = 0; i < labels.size(); i++) {
                    childrenLabels[clusterer.clusterInstance(transposed.instance(i))].add(labels.get(i));
                }
            } catch (Exception ex) {
                Logger.getLogger(HierarchyBuilder.class.getName()).log(Level.SEVERE, null, ex);
            }
        }


        return childrenLabels;
    }

    private ArrayList<String>[] randomPartitioning(int partitions, List<String> labels) {
        ArrayList<String>[] childrenLabels = new ArrayList[partitions];
        for (int i = 0; i < partitions; i++) {
            childrenLabels[i] = new ArrayList<String>();
        }

        Random rnd = new Random();
        while (!labels.isEmpty()) {
            for (int i = 0; i < partitions; i++) {
                if (labels.isEmpty()) {
                    break;
                }
                String rndLabel = labels.remove(rnd.nextInt(labels.size()));
                childrenLabels[i].add(rndLabel);
            }
        }
        return childrenLabels;
    }

    /**
     * Creates the hierarchical dataset according to the original multilabel
     * instances object and the constructed label hierarchy
     *
     * @param mlData the original multilabel instances
     * @param metaData the metadata of the constructed label hierarchy
     * @return the produced dataset
     * @throws InvalidDataFormatException In case of unexpected data format, checked exception due to instantiated MutlilabelInstances.
     */
    public static MultiLabelInstances createHierarchicalDataset(MultiLabelInstances mlData, LabelsMetaData metaData) throws InvalidDataFormatException {
        Set<String> leafLabels = mlData.getLabelsMetaData().getLabelNames();
        Set<String> metaLabels = metaData.getLabelNames();
        for (String string : leafLabels) {
            metaLabels.remove(string);
        }
        Instances dataSet = mlData.getDataSet();
        int numMetaLabels = metaLabels.size();

        // copy existing attributes
        ArrayList<Attribute> atts = new ArrayList<Attribute>(dataSet.numAttributes() + numMetaLabels);
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            atts.add(dataSet.attribute(i));
        }

        ArrayList<String>  labelValues = new ArrayList<String> ();
        labelValues.add("0");
        labelValues.add("1");

        // add metalabel attributes
        for (String metaLabel : metaLabels) {
            atts.add(new Attribute(metaLabel, labelValues));
        }

        // initialize dataset
        Instances newDataSet = new Instances("hierarchical", atts, dataSet.numInstances());

        // copy features and labels, set metalabels
        for (int i = 0; i < dataSet.numInstances(); i++) {
            //System.out.println("Constructing instance " + (i+1) + "/"  + dataSet.numInstances());
            // initialize new values
            double[] newValues = new double[newDataSet.numAttributes()];
            Arrays.fill(newValues, 0);

            // copy features and labels
            double[] values = dataSet.instance(i).toDoubleArray();
            System.arraycopy(values, 0, newValues, 0, values.length);

            // set metalabels
            for (String label : leafLabels) {
                Attribute att = dataSet.attribute(label);
                if (att.value((int) dataSet.instance(i).value(att)).equals("1")) {
                    //System.out.println(label);
                    //System.out.println(Arrays.toString(metaData.getLabelNames().toArray()));
                    LabelNode currentNode = metaData.getLabelNode(label);
                    // put 1 all the way up to the root, unless you see a 1, in which case stop
                    while (currentNode.hasParent()) {
                        currentNode = currentNode.getParent();
                        Attribute currentAtt = newDataSet.attribute(currentNode.getName());
                        // change the following to refer to the array
                        if (newValues[atts.indexOf(currentAtt)] == 1) // no need to go more up
                        {
                            break;
                        } else // put 1
                        {
                            newValues[atts.indexOf(currentAtt)] = 1;
                        }
                    }
                }
            }
            Instance instance = dataSet.instance(i);
            newDataSet.add(DataUtils.createInstance(instance, instance.weight(), newValues));
        }
        return new MultiLabelInstances(newDataSet, metaData);
    }

    private void saveToArffFile(Instances dataSet, File file) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataSet);
        saver.setFile(file);
        saver.writeBatch();
    }

    private void createXMLFile(LabelsMetaData metaData) throws Exception {
        DocumentBuilderFactory docBF = DocumentBuilderFactory.newInstance();
        DocumentBuilder docBuilder = docBF.newDocumentBuilder();
        labelsXMLDoc = docBuilder.newDocument();

        Element rootElement = labelsXMLDoc.createElement("labels");
        rootElement.setAttribute("xmlns", "http://mulan.sourceforge.net/labels");
        labelsXMLDoc.appendChild(rootElement);
        for (LabelNode rootLabel : metaData.getRootLabels()) {
            Element newLabelElem = labelsXMLDoc.createElement("label");
            newLabelElem.setAttribute("name", rootLabel.getName());
            appendElement(newLabelElem, rootLabel);
            rootElement.appendChild(newLabelElem);
        }
    }

    private void saveToXMLFile(String fileName) {
        Source source = new DOMSource(labelsXMLDoc);
        File xmlFile = new File(fileName);
        StreamResult result = new StreamResult(xmlFile);
        try {
            Transformer transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
            transformer.setOutputProperty(OutputKeys.METHOD, "xml");
            transformer.transform(source, result);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void appendElement(Element labelElem, LabelNode labelNode) {
        for (LabelNode childNode : labelNode.getChildren()) {
            Element newLabelElem = labelsXMLDoc.createElement("label");
            newLabelElem.setAttribute("name", childNode.getName());
            appendElement(newLabelElem, childNode);
            labelElem.appendChild(newLabelElem);
        }
    }

    /**
     * The different types of distributing labels to children nodes
     */
    public enum Method {
        /** random balanced distribution of labels */
        Random,
        /** distribution based on label similarity */ 
        Clustering,
        /** balanced distribution based on label similarity */
        BalancedClustering
    }
}