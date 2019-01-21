#pragma once
// Command line parsing

#include <algorithm>
#include <cassert>
#include <exception>
#include <functional>
#include <initializer_list>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "utils.h"

/* Command line view.
 * Nicer interface to represent (argc, argv).
 */
class CommandLineView {
public:
	CommandLineView (int argc, const char * const * argv) : argc_ (argc), argv_ (argv) { assert (argv != nullptr); }

	// Raw access
	string_view program_name () const { return {argv_[0]}; }
	std::size_t nb_arguments () const noexcept { return argc_ - 1; }
	string_view argument (std::size_t index) const { return {argv_[index + 1]}; }

private:
	int argc_;
	const char * const * argv_;
};

/* Command line argument parser.
 *
 * Declare flags, options with values and positional arguments.
 * Provide callback functions that will be called when the options are detected (actions).
 * Then call parse, which will parse the entire command line and trigger callbacks.
 * Also generate a usage message from the list of options.
 */
class CommandLineParser {
public:
	class Exception;

	/* Declare options.
	 * Throws Exception if no names are given for named options.
	 * 'action' is the function to call when the option is discovered.
	 * Supported option names can contain anything else than '='.
	 * Due to shell parsing, users should only use alphanumeric and '-' '_' to avoid problems.
	 *
	 * Usage text:
	 * 'description' should fit on one line (very short text).
	 * 'value_name' is the name of the value pattern.
	 */
	void flag (std::initializer_list<string_view> names, string_view description, std::function<void()> action);
	void option (std::initializer_list<string_view> names, string_view value_name, string_view description,
	             std::function<void(string_view value)> action);
	void option2 (std::initializer_list<string_view> names, string_view value1_name, string_view value2_name,
	              string_view description, std::function<void(string_view value1, string_view value2)> action);

	/* Declare positional arguments.
	 * Each call of add_positional_argument declares a single argument.
	 * Arguments declared that way are mandatory (all must be present).
	 * They are parsed in the order of declaration.
	 * Usage text is the same as for options.
	 */
	void positional (string_view value_name, string_view description, std::function<void(string_view value)> action);

	/* Print usage to output.
	 * Options are given in the order of declaration.
	 * Options names are listed lexicographically.
	 */
	void usage (std::FILE * output, string_view program_name) const;

	/* Parse arguments in order of appearance on the command line.
	 *
	 * For each flag or option, call the appropriate callback each time it is discovered.
	 * Anything starting with '-' is considered an option, and must match a defined option name.
	 * Options can use single or dual dashes ('-' / '--') even for short options (one letter).
	 * Value options will consume data after an equal sign ('-f=value'), or the next argument.
	 * The value is given to the action callback.
	 *
	 * All elements which are not option-like are considered positional arguments.
	 * The number of all found positional argument must match the declared ones.
	 * Values are given to the callbacks in order of declaration.
	 *
	 * Option '--' stops option parsing: next arguments are always considered positional.
	 * A parser with no options declared starts as if '--' was passed before any user argument.
	 *
	 * Throws Exception if a parsing error occurs.
	 */
	void parse (const CommandLineView & command_line);

private:
	/* Options are stored in a vector, and indexed by name for quick search
	 *
	 * Use std::map as we search with a string_view, and only map supports template keys.
	 * An unordered_map would create a std::string everytime a comparison is made.
	 */
	struct Option {
		enum class Type {
			Flag,  // Option is an optional flag without value
			Value, // Option is an optional flag with a value
			Value2 // Option is an optional flag with two values
		};
		Type type;

		std::function<void()> flag_action;
		std::function<void(string_view value)> value_action;
		std::function<void(string_view value1, string_view value2)> value2_action;

		// Usage text (may be null)
		std::string value_name;
		std::string value2_name;
		std::string description;
	};
	struct OptionNameLess : std::less<void> {
		// Ordering of option names.
		// Smaller length first, then lexicographic.
		template <typename L, typename R> bool operator() (const L & lhs, const R & rhs) const {
			return lhs.size () < rhs.size () || (lhs.size () == rhs.size () && std::less<void>::operator() (lhs, rhs));
		}
	};
	std::map<std::string, int, OptionNameLess> option_index_by_name_;
	std::vector<Option> options_;

	struct PositionalArgument {
		std::function<void(string_view value)> action;
		std::string value_name;
		std::string description;
	};
	std::vector<PositionalArgument> positional_arguments_;

	Option & new_named_uninitialized_option (std::initializer_list<string_view> names);
};

class CommandLineParser::Exception : public std::exception {
private:
	std::string message_;

public:
	Exception (string_view message) : message_ (to_string (message)) {}
	Exception (std::string && message) : message_ (std::move (message)) {}
	Exception (const char * message) : message_ (message) {}
	const char * what () const noexcept final { return message_.c_str (); }
};

/******************************************************************************
 * Impl.
 */
inline CommandLineParser::Option &
CommandLineParser::new_named_uninitialized_option (std::initializer_list<string_view> names) {
	if (names.size () == 0) {
		throw Exception ("Declaring option with no name");
	}
	auto index = static_cast<int> (options_.size ());
	options_.emplace_back ();
	for (auto name : names) {
		if (empty (name)) {
			throw Exception ("Empty option name declaration");
		}
		auto r = option_index_by_name_.emplace (to_string (name), index);
		if (!r.second) {
			throw Exception (fmt::format ("Option '{}' has already been declared", name));
		}
	}
	return options_.back ();
}

inline void CommandLineParser::flag (std::initializer_list<string_view> names, string_view description,
                                     std::function<void()> action) {
	auto & opt = new_named_uninitialized_option (names);
	opt.type = Option::Type::Flag;
	opt.flag_action = std::move (action);
	opt.description = to_string (description);
}

inline void CommandLineParser::option (std::initializer_list<string_view> names, string_view value_name,
                                       string_view description, std::function<void(string_view value)> action) {
	auto & opt = new_named_uninitialized_option (names);
	opt.type = Option::Type::Value;
	opt.value_action = std::move (action);
	opt.value_name = to_string (value_name);
	opt.description = to_string (description);
}

inline void CommandLineParser::option2 (std::initializer_list<string_view> names, string_view value1_name,
                                        string_view value2_name, string_view description,
                                        std::function<void(string_view value1, string_view value2)> action) {
	auto & opt = new_named_uninitialized_option (names);
	opt.type = Option::Type::Value2;
	opt.value2_action = std::move (action);
	opt.value_name = to_string (value1_name);
	opt.value2_name = to_string (value2_name);
	opt.description = to_string (description);
}

inline void CommandLineParser::positional (string_view value_name, string_view description,
                                           std::function<void(string_view value)> action) {
	positional_arguments_.emplace_back ();
	auto & arg = positional_arguments_.back ();
	arg.action = std::move (action);
	arg.value_name = to_string (value_name);
	arg.description = to_string (description);
}

inline void CommandLineParser::usage (std::FILE * output, string_view program_name) const {
	assert (output != nullptr);
	std::size_t description_text_offset = 0;

	const bool has_options = !option_index_by_name_.empty ();
	const bool has_positional_arguments = !positional_arguments_.empty ();

	std::vector<std::string> option_text (options_.size ());
	std::string usage_line_pos_arg_text;

	if (has_options) {
		assert (!options_.empty ());
		// List of options : create option text
		for (const auto & e : option_index_by_name_) {
			// Append option name to option line
			auto & text = option_text[e.second];
			const auto & opt_name = e.first;
			if (!text.empty ()) {
				text.append (", ");
			}
			text.append (opt_name.size () == 1 ? "-" : "--");
			text.append (opt_name);
		}
		// Append value name for value options
		for (std::size_t i = 0; i < options_.size (); ++i) {
			const auto & opt = options_[i];
			if (opt.type == Option::Type::Value) {
				option_text[i] += fmt::format (" <{}>", opt.value_name);
			} else if (opt.type == Option::Type::Value2) {
				option_text[i] += fmt::format (" <{}> <{}>", opt.value_name, opt.value2_name);
			}
		}

		// Get len for alignment of description text
		auto max_option_text_len_it =
		    std::max_element (option_text.begin (), option_text.end (),
		                      [](const std::string & lhs, const std::string & rhs) { return lhs.size () < rhs.size (); });
		description_text_offset = std::max (description_text_offset, max_option_text_len_it->size ());
	}

	if (has_positional_arguments) {
		for (const auto & pos_arg : positional_arguments_) {
			usage_line_pos_arg_text.append (" ");
			usage_line_pos_arg_text.append (pos_arg.value_name);
			description_text_offset = std::max (description_text_offset, pos_arg.value_name.size ());
		}
	}

	fmt::print (output, "Usage: {}{}{}\n", program_name, has_options ? " [options]" : "", usage_line_pos_arg_text);

	auto print_opt_arg_line = [&output, &description_text_offset](string_view name, string_view description) {
		fmt::print (output, "  {0: <{1}}  {2}\n", name, description_text_offset, description);
	};

	if (has_positional_arguments) {
		fmt::print (output, "\nArguments:\n");
		for (const auto & pos_arg : positional_arguments_) {
			print_opt_arg_line (pos_arg.value_name, pos_arg.description);
		}
	}
	if (has_options) {
		fmt::print (output, "\nOptions:\n");
		print_opt_arg_line ("--", "Disable option parsing after this");
		for (std::size_t i = 0; i < options_.size (); ++i) {
			print_opt_arg_line (option_text[i], options_[i].description);
		}
	}
}

inline string_view make_string_view (std::cmatch::const_reference match_result) {
	return make_string_view (match_result.first, match_result.second);
}

inline void CommandLineParser::parse (const CommandLineView & command_line) {
	std::size_t current = 0;
	const std::size_t nb = command_line.nb_arguments ();

	std::size_t nb_pos_arg_seen = 0;
	const std::size_t nb_pos_arg_needed = positional_arguments_.size ();

	bool option_parsing_enabled = !option_index_by_name_.empty ();

	// Option is: one or two dashes, then option name, then '=<value>' or nothing
	std::regex option_name_regex{"^(--|-)([^=]+)(=|$)"};
	assert (option_name_regex.mark_count () == 3);
	std::cmatch matched_positions;

	while (current < nb) {
		auto arg = command_line.argument (current);

		if (option_parsing_enabled && starts_with ('-', arg)) {
			if (arg == "--") {
				// Special case, consider everything after that as positional arguments
				option_parsing_enabled = false;
			} else {
				// Argument is an option
				if (!std::regex_search (arg.begin (), arg.end (), matched_positions, option_name_regex)) {
					throw Exception (fmt::format ("Bad format: option '{}'", arg));
				}
				assert (matched_positions.ready ());
				assert (matched_positions.size () == 4);
				auto dashes = make_string_view (matched_positions[1]);
				auto opt_name = make_string_view (matched_positions[2]);
				auto equal_sign_or_empty = make_string_view (matched_positions[3]);
				auto opt_name_with_dashes = make_string_view (dashes.begin (), opt_name.end ());

				auto it = option_index_by_name_.find (opt_name);
				if (it == option_index_by_name_.end ()) {
					throw Exception (fmt::format ("Unknown option '{}'", opt_name_with_dashes));
				}
				Option & opt = options_[static_cast<std::size_t> (it->second)];

				if (opt.type == Option::Type::Flag) {
					// Simple flag option
					if (!empty (equal_sign_or_empty)) {
						throw Exception (fmt::format ("Flag '{}' takes no value", opt_name_with_dashes));
					}
					assert (opt.flag_action);
					opt.flag_action ();
				} else {
					// Value option, extract value
					string_view value;
					if (!empty (equal_sign_or_empty)) {
						// Value is the rest of the argument
						value = make_string_view (equal_sign_or_empty.end (), arg.end ());
					} else {
						// Value is the next argument
						++current;
						if (current == nb) {
							throw Exception (fmt::format ("Option '{}' requires a value: {}", opt_name_with_dashes, opt.value_name));
						}
						value = command_line.argument (current);
					}
					if (opt.type == Option::Type::Value) {
						assert (opt.value_action);
						opt.value_action (value);
					} else if (opt.type == Option::Type::Value2) {
						// Extract second value
						++current;
						if (current == nb) {
							throw Exception (
							    fmt::format ("Option '{}' requires a second value: {}", opt_name_with_dashes, opt.value2_name));
						}
						const string_view value2 = command_line.argument (current);
						assert (opt.value2_action);
						opt.value2_action (value, value2);
					}
				}
			}
		} else {
			// Argument is a positional argument
			if (nb_pos_arg_seen == nb_pos_arg_needed) {
				throw Exception (fmt::format ("Unexpected argument '{}' at position {}: requires {} arguments", arg,
				                              nb_pos_arg_seen, nb_pos_arg_needed));
			}
			auto & pos_arg = positional_arguments_[nb_pos_arg_seen];
			assert (pos_arg.action);
			pos_arg.action (arg);
			++nb_pos_arg_seen;
		}
		++current;
	}

	if (nb_pos_arg_seen != nb_pos_arg_needed) {
		throw Exception (
		    fmt::format ("Only {} positional arguments given, {} requested", nb_pos_arg_seen, nb_pos_arg_needed));
	}
}
