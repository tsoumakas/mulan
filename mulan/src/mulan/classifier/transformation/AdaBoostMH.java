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
package mulan.classifier.transformation;

import weka.classifiers.meta.AdaBoostM1;

/**
 * <p>Implementation of the AdaBoost.MH algorithm based on Weka's AdaBoostM1.
 * </p><p>For more information, see <em>Schapire, R.E.; Singer, Y. (2000).
 * BoosTexter: A boosting-based system for text categorization. Machine
 * Learning. 39(2/3):135-168.</em> </p>
 *
 * @author Grigorios Tsoumakas
 * @version 2013.01.22
 */
public class AdaBoostMH extends IncludeLabelsClassifier {

    /**
     * Default constructor
     */
    public AdaBoostMH() {
        super(new AdaBoostM1());
    }
}