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
 *    AdaBoostMH.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.TechnicalInformation;

/**
 <!-- globalinfo-start -->
 * Implementation of the AdaBoost.MH algorithm based on Weka's AdaBoostM1. For more information, see<br/>
 * <br/>
 * Robert E. Schapire, Yoram Singer (2000). BoosTexter: A boosting-based system for text categorization. Machine Learning. 39(2/3):135-168.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Schapire2000,
 *    author = {Robert E. Schapire and Yoram Singer},
 *    journal = {Machine Learning},
 *    number = {2/3},
 *    pages = {135-168},
 *    title = {BoosTexter: A boosting-based system for text categorization},
 *    volume = {39},
 *    year = {2000}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class AdaBoostMH extends IncludeLabelsClassifier {

    /**
     * Default constructor
     */
    public AdaBoostMH() {
        super(new AdaBoostM1());
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR,
                "Robert E. Schapire and Yoram Singer");
        result.setValue(TechnicalInformation.Field.TITLE,
                "BoosTexter: A boosting-based system for text categorization");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Machine Learning");
        result.setValue(TechnicalInformation.Field.YEAR, "2000");
        result.setValue(TechnicalInformation.Field.PAGES, "135-168");
        result.setValue(TechnicalInformation.Field.VOLUME, "39");
        result.setValue(TechnicalInformation.Field.NUMBER, "2/3");
        return result;
    }

    /**
     * Returns a string describing the classifier.
     *
     * @return a string description of the classifier
     */
    @Override
    public String globalInfo() {
        return "Implementation of the AdaBoost.MH algorithm based on Weka's "
                + "AdaBoostM1. For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

}