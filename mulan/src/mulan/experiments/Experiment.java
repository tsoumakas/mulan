package mulan.experiments;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 * Abstract class for all experiments
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 */
public abstract class Experiment implements TechnicalInformationHandler{

    /**
     * Gets the {@link TechnicalInformation} for the current Experiment.
     *
     * @return technical information
     */
    public abstract TechnicalInformation getTechnicalInformation();
}
