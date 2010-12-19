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

package mulan.console;

import java.io.File;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

/**
 * Helper for construction of common command line option reused by multiple classes.
 * 
 * @author Jozef Vilce
 */
public final class CommonOptions {

	/** Seed option short name */
	public static final String SEED_OPT = "s";
	/** Help option short name */
	public static final String HELP_OPT = "h";
	/** Train data set option short name */
	public static final String TRAIN_DATA_OPT = "trd";
	/** Test data set option short name */
	public static final String TEST_DATA_OPT = "tsd";
	/** Labels definition option short name */
	public static final String LABELS_DEF_DATA_OPT = "ld";
	
	/**
	 * Creates option for specifying train data set
	 * @return
	 */
	@SuppressWarnings("static-access")
	public static Option getTrainDataOption(){
		return OptionBuilder.withLongOpt("train-data").hasArg().withType(File.class)
			.withDescription("Path to the data set which should be used for training a learner. The option is required.")
			.create(TRAIN_DATA_OPT);
	}
	
	/**
	 * Creates option for specifying test data set
	 * @return
	 */
	@SuppressWarnings("static-access")
	public static Option getTestDataOption(){
		return OptionBuilder.withLongOpt("test-data").hasArg().withType(File.class)
			.withDescription("Path to the data set which should be used for testing a learner. The option is required.")
			.create(TEST_DATA_OPT);
	}
	
	/**
	 * Creates option for specifying labels definition file for data sets
	 * @return
	 */
	@SuppressWarnings("static-access")
	public static Option getLabelsDefinitionOption(){
		return OptionBuilder.withLongOpt("labels-definition").hasArg().withType(File.class)
			.withDescription("Path to the XML file with label definitions. The option is required.")
			.create(LABELS_DEF_DATA_OPT);
	}
	
	/**
	 * Creates option for showing the help
	 * @return
	 */
	@SuppressWarnings("static-access")
	public static Option getHelpOption(){
		return OptionBuilder.withLongOpt("help").create(HELP_OPT);
	}
	
	/**
	 * Creates option for specifying a randomness seed value
	 * @return
	 */
	@SuppressWarnings("static-access")
	public static Option getSeedOption(){
		return OptionBuilder.withLongOpt("seed").hasArg().withType(Number.class)
					.withDescription("Optional seed value for randomness.").create(SEED_OPT);
	}
}
