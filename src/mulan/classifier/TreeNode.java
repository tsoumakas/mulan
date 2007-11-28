package mulan.classifier;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
/*
 * TreeNode.java
 *
 * Created on 28 Ιούνιος 2007, 11:05 πμ
 */

/**
 *
 * @author greg
 */
public class TreeNode {
    private int id;
    private Instances header;
    private Classifier classifier;
    ArrayList<String> labels;
    ArrayList<TreeNode> children;
    
    /** Creates a new instance of TreeNode */
    public TreeNode() {
        classifier = new weka.classifiers.bayes.NaiveBayes();
    }
    
    public void setId(int x)
    {
        id = x;
    }
    
    public void saveClassifier()
    {
        try {
            weka.core.SerializationHelper.write(id + ".model", classifier);
            classifier = null;
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void loadClassifier()
    {
        try {
            classifier = (Classifier) weka.core.SerializationHelper.read(id + ".model");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    
    public double classifyInstance(Instance instance) throws Exception
    {
       // if (classifier == null)
       //     loadClassifier();
        
        double result;
        result = classifier.classifyInstance(instance);

        /*
        try {
            result = classifier.classifyInstance(instance);
            classifier = null;
        } catch (Exception ex) {
            ex.printStackTrace();
            result = -1;
        }
         */
        
        return result;
    }
    
    public Instances getHeader() 
    {
        return header;
    }
    
    public void setHeader(Instances h)
    {
        header = new Instances(h);
    }

    public void setLabels(ArrayList<String> l) 
    {
        labels = new ArrayList<String>();
        for (int i=0; i<l.size(); i++)
            labels.add(l.get(i));
    }
    
    public void setClassifier(Classifier c) throws Exception
    {
        classifier = Classifier.makeCopy(c);
    }
    
    public void buildClassifier(Instances data) throws Exception
    {
        classifier.buildClassifier(data);
    }
}
