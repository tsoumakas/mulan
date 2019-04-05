package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.BinaryRelevance;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.TechnicalInformation;

public abstract class HomogeneousEnsembleMultiLabelLearner extends MultiLabelLearnerBase{

	/** The base multi-label learner.
    */
    protected MultiLabelLearner baseMlLearner;
   

/**
    * Creates a new instance of {@link EnsembleBasedMultiLabelLeaner}}
    * with default {@link BinaryRelevance} multi-label learner
    */
   public HomogeneousEnsembleMultiLabelLearner() {
       this(new BinaryRelevance(new J48()));
   }

   /**
    * Creates a new instance.
    *
    * @param baseMlLearner the base classifier which will be used internally
    * to handle the data.
    * @see Classifier
    */
   public HomogeneousEnsembleMultiLabelLearner(MultiLabelLearner baseMlLearner) {
       this.baseMlLearner = baseMlLearner;
   }

   /**
    * Returns the {@link MultiLabelLearner} which is used internally by the ensemble model.
    *
    * @return the internally used multi-label learner
    */
   public MultiLabelLearner getBaseLearner() {
       return baseMlLearner;
   }
   
   /**
	 * @param baseMlLearner the baseMlLearner to set
     */
   public void setBaseMlLearner(MultiLabelLearner baseMlLearner) {
	   this.baseMlLearner = baseMlLearner;
   }

   /**
    * Returns an instance of a TechnicalInformation object, containing detailed
    * information about the technical background of this class, e.g., paper
    * reference or book this class is based on.
    *
    * @return the technical information about this class
    */
   public TechnicalInformation getTechnicalInformation() {
	   return null;
   }

   /**
    * Returns a string describing the classifier.
    *
    * @return a string description of the classifier
    */
   public String globalInfo() {
       return "Base class for multi-label learners, which build an ensemble of multi-label leaners "
               + "For more information, see\n\n"
               + getTechnicalInformation().toString();
   }

}
