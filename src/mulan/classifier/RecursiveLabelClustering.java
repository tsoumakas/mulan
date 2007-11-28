/*
 * RecursiveLabelClustering.java
 *
 * Created on 23 Ιούνιος 2007, 9:08 μμ
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package mulan.classifier;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.clusterers.BalancedKMeans;
import weka.clusterers.ConstrainedKMeans;
import weka.clusterers.DBScan;
import weka.clusterers.OPTICS;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.XMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
import weka.experiment.ClassifierSplitEvaluator;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author greg
 */
public class RecursiveLabelClustering extends AbstractMultiLabelClassifier {
    private int nodesFired;
    private int nodeCount = 0;
    private int maxElements=15;
    private int numClusters=10;
    private Instances binaryHeader;
    private Instances data;
    private Instances tagData;
    private TreeNode root;
    private ArrayList<String> trueLabels;
    private ArrayList<String> labelsAll;
    
    public static class ClassifierTree {        
        Classifier[] classifier;
        ClassifierTree[] children; 
    }
    
    /**
     * Creates a new instance of RecursiveLabelClustering
     */
    
    public void setNumClusters(int x)
    {
        numClusters = x;
    }
    
    public void setMaxElements(int x)
    {
        maxElements = x;
    }
    
    public RecursiveLabelClustering() {
    }    

    public ArrayList<String> getLabels(Instances data) throws Exception {
        ArrayList<String> labels = new ArrayList<String>();
                
        for (int i=0; i<numLabels; i++)
            labels.add(data.attribute(data.numAttributes()-numLabels+i).name());
            
        return labels;
    }
    
    private Instances createLabelData(ArrayList<String> labels) throws Exception {
        Instances newInstances = new Instances(tagData, 0);
        
        //* add sequentially
        for (int i=0; i<labels.size(); i++)
        {
            for (int j=0; j<tagData.numInstances(); j++)
            {
                if (labels.get(i).equals(tagData.instance(j).attribute(0).value((int) tagData.instance(j).value(0))))
                {
                    newInstances.add(tagData.instance(j));
                    continue;
                }
            }
        }

        /* add randomly
        for (int i=0; i<tagData.numInstances(); i++)
            if (labels.contains(tagData.instance(i).attribute(0).value((int) tagData.instance(i).value(0))))
                newInstances.add(tagData.instance(i));
        */
        
        Remove rm = new Remove();
        int[] indices = {0};
        rm.setAttributeIndicesArray(indices);
        rm.setInputFormat(newInstances);
        newInstances = Filter.useFilter(newInstances, rm);
        
        // convert to numeric
        FastVector Atts = new FastVector();
        for (int i=0; i<newInstances.numAttributes(); i++)
        {
            Attribute Att = new Attribute("numeric" + i);
            Atts.addElement(Att);
        }
        Instances newInstances2 = new Instances("data",Atts,newInstances.numInstances());
        for (int i=0; i<newInstances.numInstances(); i++)
            newInstances2.add(newInstances.instance(i));
        
        return newInstances2;
    }
    
    private Instances createData(ArrayList<String> labels) throws Exception {
        Remove rm = new Remove();        
        
        String indices = "";
        indices = indices + (data.numAttributes()-numLabels+2) +"-" + data.numAttributes();
        rm.setAttributeIndices(indices);
        rm.setInputFormat(data);
        Instances newInstances = Filter.useFilter(data, rm);
        
        // 
        int numAttributes = data.numAttributes();
        int numInstances = newInstances.numInstances();
        for (int i=0; i<numInstances; i++) 
        {
            newInstances.instance(i).setValue(newInstances.numAttributes()-1, 0);
         
            
            for (int j=0; j<numLabels; j++) {
                if ((data.instance(i).value(numAttributes-numLabels+j) == 1) &&
                    (labels.contains(data.instance(i).attribute(numAttributes-numLabels+j).name())))
                    newInstances.instance(i).setValue(newInstances.numAttributes()-1, 1);            
            }
        }
        
        newInstances.setClassIndex(newInstances.numAttributes()-1);

        /*
        DataSink ds = new DataSink("c:/work/datasets/delicious/xxx.arff");
        ds.write(newInstances);
        //System.out.println(newInstances);
        //*/

        
        return newInstances;
    }    
    
    Instances reverseFullDatasetAndKeepAttributesIns(Instances data) throws Exception
    {
        String tempFilename = "reversed.arff";
        FileWriter fw = new FileWriter(new File(tempFilename));
        int numAtt = data.numAttributes();
        int numIns = data.numInstances();

        fw.write("@relation reversed\n\n");
        fw.write("@attribute tag_name string\n");
        for(int j=0; j<numIns; j++)
        {
            fw.write("@attribute "+j+" {0,1}\n");
        }
        fw.write("\n@data \n");

        for(int i=numAtt-numLabels; i<numAtt; i++)
        {
            fw.write(data.attribute(i).name()+",");
            double[] r = data.attributeToDoubleArray(i);
            for(int t=0; t<r.length-1; t++)
               fw.write((int)r[t]+",");
            fw.write((int)r[r.length-1]+"\n");
        }
        fw.close();
        
        ArffLoader l = new ArffLoader();
        l.setSource(new File(tempFilename));
        Instances dataReturn = l.getDataSet();
        
        return dataReturn;
    }

    
    public void buildClassifier(Instances data) throws Exception {
        this.data = data;
        
        // get labels        
        labelsAll = getLabels(data);
        //System.out.println(Arrays.toString(labels.toArray()));
        
        // keep labels only and reverse
        tagData = reverseFullDatasetAndKeepAttributesIns(data);
        //System.out.println("tagData:\n" + tagData.toString());

        root = buildTree(labelsAll);
        System.out.println("Total Nodes: " + nodeCount);
    }   
    
        
    public TreeNode buildTree(ArrayList<String> labels) throws Exception {
        //System.out.println(Arrays.toString(labels.toArray()));        
        
        nodeCount++;
        TreeNode aNode = new TreeNode();
        aNode.setLabels(labels);
        aNode.setId(nodeCount);

        // create training data from labels in this node
        Instances nodeData = createData(labels);
        // System.out.println("nodeData: \n" + nodeData.toString());

        // build binary classifier for labels in this node
        aNode.buildClassifier(nodeData);
        //aNode.saveClassifier();
        
        // keep the header
        nodeData = new Instances(nodeData, 0);
        //aNode.setHeader(nodeData);
        
        // keep the header once
        if (labels.size() == numLabels)
        {
            binaryHeader = new Instances(nodeData, 0);
        }
        
        // if single label then return the single label classifier
        if (labels.size() == 1)
        {
           System.out.println(Arrays.toString(labels.toArray()));                   
           return aNode;
        }
        
        // if labels less than a threshold build a single label classifier for each child
        if (labels.size() <= maxElements)
        {
            aNode.children = new ArrayList<TreeNode>();
            for (int i=0; i<labels.size(); i++)
            {
                ArrayList<String> childLabels = new ArrayList<String>();
                childLabels.add(labels.get(i));
                TreeNode aChild = buildTree(childLabels);
                aNode.children.add(aChild);
            }
            return aNode;
        }
        
        // if labels more than a threshold then cluster and recurse for each cluster
        if (labels.size() > maxElements) 
        {

            // prepare data for clustering
            Instances nodeLabelData = createLabelData(labels);
               
            /*            
            DBScan clusterer = new DBScan();
            clusterer.setMinPoints(10);
            clusterer.buildClusterer(nodeLabelData);
            numClusters = clusterer.numberOfClusters();
            //*/
            //*            
            // cluster using k-means
            //SimpleKMeans clusterer = new SimpleKMeans();
            ConstrainedKMeans clusterer = new ConstrainedKMeans();
            clusterer.setMaxIterations(20);
            clusterer.setNumClusters(numClusters);
            clusterer.buildClusterer(nodeLabelData);
            //*/
            /*
            XMeans clusterer = new XMeans();
            clusterer.setMinNumClusters(2);
            clusterer.setMaxNumClusters(10);
            clusterer.buildClusterer(tagData);
            numClusters = clusterer.numberOfClusters();
            //*/            
            //System.out.println("FINISHED CLUSTERING!");
            
            ArrayList[] childrenLabels = new ArrayList[numClusters];
            for (int i=0; i<numClusters; i++)
                childrenLabels[i] = new ArrayList<String>();            
            for (int i=0; i<nodeLabelData.numInstances(); i++) 
                childrenLabels[clusterer.clusterInstance(nodeLabelData.instance(i))].add(labels.get(i));

            for (int i=0; i<numClusters; i++)
            {
                System.out.println("Cluster " + i);
                System.out.println(Arrays.toString(childrenLabels[i].toArray()));
            }

            /* 
            // write some dumm/sequential code to balance the clusters            
            int properSize = (int) Math.ceil(nodeLabelData.numInstances() / (double) numClusters);
            System.out.println("proper size=" + properSize);

            for (int i=0; i<numClusters; i++)
                while (childrenLabels[i].size() > properSize)
                {       
                    String label = (String) childrenLabels[i].remove(0);
                    for (int j=0; j<numClusters; j++)
                    {
                        if (childrenLabels[j].size() < properSize)
                        {
                            childrenLabels[j].add(label);
                            break;
                        }
                    }
                }
            //*/ 

            // delete data and clusterer
            nodeLabelData = null;
            clusterer = null;
            System.gc();
            
            
            aNode.children = new ArrayList<TreeNode>();
            for (int i=0; i<numClusters; i++)
            {
                if (childrenLabels[i].size() == labels.size())
                {
                    for (int j=0; j<labels.size(); j++)
                    {
                        ArrayList<String> temp = new ArrayList<String>();
                        temp.add(labels.get(j));
                        TreeNode aChild = buildTree(temp);
                        aNode.children.add(aChild);
                    }
                    return aNode;
                }
                
                if (childrenLabels[i].size() > 0)
                {
                    TreeNode aChild = buildTree(childrenLabels[i]);
                    aNode.children.add(aChild);
                }
            }
            return aNode;
        }        
        
        return null;
    }
    
    private Instances createClusterDataset(Instances data, ArrayList<String> clusterTags) throws Exception {
        Remove rm = new Remove();        
        
        String indices = "";
        indices = indices + (data.numAttributes()-numLabels+1) +"-" + data.numAttributes();
        rm.setAttributeIndices(indices);
        rm.setInputFormat(data);
        Instances newInstances = Filter.useFilter(data, rm);
        
        //DataSink ds = new DataSink("c:/work/datasets/delicious/xxx.arff");
        //ds.write(newInstances);
        //System.out.println(newInstances);
        
        
        // 
        int numAttributes = data.numAttributes();
        int numInstances = newInstances.numInstances();
        for (int i=0; i<numInstances; i++) 
        {
            newInstances.instance(i).setValue(newInstances.numAttributes(), 0);
            
            for (int j=0; j<numLabels; j++)
                if ((data.instance(i).value(numAttributes-numLabels+j) == 1) &&
                    (clusterTags.contains(data.attribute(numAttributes-numLabels+j).name())))
                    newInstances.instance(i).setValue(newInstances.numAttributes(), 1);            
        }
        
        return newInstances;
    }

    private void nodePrediction(TreeNode node, Instance instance) throws Exception {
        //System.out.println(Arrays.toString(node.labels.toArray()));
        
        nodesFired++;
        
        //instance.setDataset(node.getHeader());        
        
        if (node.classifyInstance(instance) == 1)
            if (node.labels.size() == 1) 
                trueLabels.add(node.labels.get(0));
            else
                for (int i=0; i<node.children.size(); i++)
                    nodePrediction(node.children.get(i), instance);    
    }
    
    protected Prediction makePrediction(Instance instance) throws Exception {
        trueLabels = new ArrayList<String>();
        ArrayList actualLabels = new ArrayList<String>();

        for (int i=0; i<numLabels; i++)
            if (instance.value(instance.numAttributes()-numLabels+i) == 1)
                actualLabels.add(instance.attribute(instance.numAttributes()-numLabels+i).name());
        
        // correct, but not sparse instance
        Instance newInstance;
        if (instance instanceof SparseInstance)
            newInstance = new SparseInstance(instance.numAttributes()-numLabels+1); 
        else
            newInstance = new Instance(instance.numAttributes()-numLabels+1);
        for (int i=0; i<newInstance.numAttributes()-1; i++)
            newInstance.setValue(i, instance.value(i));
        newInstance.setDataset(binaryHeader);
        
        nodesFired = 0;
        for (int i=0; i<root.children.size(); i++)
            nodePrediction(root.children.get(i), newInstance);
        
        double predictions[] = new double[numLabels];
        double confidences[] = new double[numLabels];

        System.out.println("Nodes fired: " + nodesFired);
        /*
        System.out.println("All labels: ");
        System.out.println(Arrays.toString(labelsAll.toArray()));
        System.out.println("Predicted labels: ");
        System.out.println(Arrays.toString(trueLabels.toArray()));
        System.out.println("Actual labels: ");
        System.out.println(Arrays.toString(actualLabels.toArray()));
        */
        for (int i = 0; i < numLabels; i++)
        {
            if (trueLabels.contains(labelsAll.get(i)))
                predictions[i] = 1;
            else
                predictions[i] = 0;
            confidences[i] = 0;
        }
        /*
        System.out.println("Predictions: ");
        System.out.println(Arrays.toString(predictions));
        */
        Prediction result = new Prediction(predictions, confidences);
        return result;
    }
    
}
