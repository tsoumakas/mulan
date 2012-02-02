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
 *    Experiment.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.experiments;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 * Abstract base class for all experiments
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @author Grigorios Tsoumakas
 * @version 2010.12.27
 */
public abstract class Experiment implements TechnicalInformationHandler {

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    abstract public TechnicalInformation getTechnicalInformation();
}