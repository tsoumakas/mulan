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
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.console.builder;

import mulan.classifier.MultiLabelLearner;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/**
 * Builder of {@link MultiLabelLearner} instance from command line interface.
 * 
 * @author Jozef Vilcek
 */
public interface LearnerBuilder {

	/**
	 * Returns command line options supported by the {@link LearnerBuilder}
	 * @return
	 */
	Options getOptions();
	
	/**
	 * Returns the {@link Class} type of the {@link MultiLabelLearner} operated by this {@link LearnerBuilder}.
	 * @return
	 */
	Class<? extends MultiLabelLearner> getSupportedType();
	
	/**
	 * Builds up the {@link MultiLabelLearner} instance from {@link CommandLine} options.
	 * 
	 * @param cmdLine the input command line parameters for learner creation
	 * @return
	 * @throws ParseException if input {@link CommandLine} parameters are not valid
	 */
	MultiLabelLearner build(CommandLine cmdLine) throws ParseException;
}
