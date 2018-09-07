# AutoMama - desktop automation framework

## About

AutoMama is an automation framework for desktops. It has several types of tasks which cover ~90% of use cases.

[![AutoMama demo](https://img.youtube.com/vi/X92b26uaH7E/0.jpg)](https://www.youtube.com/watch?v=X92b26uaH7E)
## How to install

1. Clone repo
2. `pip install -r requirements`

## How to run

`python visual <path_to_workflow>.yml`

## How to write workflow?


Types of actions and required arguments:
### `config`
Set default values for some of the variables. If used - should be on the first place in the instruction list.

- `between_sleep`: [optional][default=1] - Sleep time in seconds between each action.
- `default_match_strategy`: [optional][default=template_matching] - default match strategy for
                                                                    clicks.

### `screenshot`
Takes screenshot and saves it in path.

- `path`: [required] - path to output image file with screenshot.


### `key_sequence`
Presses key sequence for example for opening spotlight command+space.

- `keys`: [required] - list of keys to pres in a sequence.
- `sleep`: [optional][default=0] - time in seconds between key "downs" and key "ups".

### `type_write`
Writing text

- `text`: [required] - text to write.
- `interval`: [optional][default=0.05] time interval in seconds between each key press.

### `click`
Search for a fragment on the screen from template and click left mouse button.

- `path`: [required] - path to template image.
- `show`: [optional][default=false] - shows the screen shot of a screen with the template matched on it. Useful if AutoMama is mis-clicking.
- `strategy`: [optional][default=template_matching] - the Computer Vision method to search for a region to click.
    - `template_matching` - more accurate but requires template size which exactly matches its size on the screen.
    - `sift` - less accurate but is scale-invariant - so the template can be in any size.

### `wait_until_visible`
Wait until the template is not visible on the screen.

- `path`: [required] - path to template image.
- `check_interval`: [optional][default=1] - time to sleep between the checks
- `strategy`: [optional][default=template_matching] - the same as for `click`
- `timeout`: [optional][default=120] - time for seconds after we assume the action failed.

### `wait_for_appear`
Wait for a template to appear.
- `path`: [required] - path to template image.
- `check_interval`: [optional][default=1] - time to sleep between the checks
- `strategy`: [optional][default=template_matching] - the same as for `click`
- `timeout`: [optional][default=120] - time for seconds after we assume the action failed.

### `scroll_for_appear`
Scrolls until the matcher can see the template on the screen.

- `path`: [required] - path to template image.
- `check_interval`: [optional][default=1] - time to sleep between the checks
- `strategy`: [optional][default=template_matching] - the same as for `click`
- `timeout`: [optional][default=120] - time for seconds after we assume the action failed.
- `distance`: [optional][default=-10] - how many "clicks" should be scrolled and which direction in every iteration.

### BETA `find_text`
Search for a text on the screen using Google OCR (the google API key is needed with Google Vision API enabled)

- `name`: [required] - human readable name for an action
- `text`: [reqired] - a text to be found on the screen
- `show`: [optional][default=false] shows the screen shot of a screen with all the text found with green rectangle for matched words and red for not matched.

## Generic Arguments for all actions:
- `name`: [required] - human readable name for an action
- `type`: [required] - type of an action.
- `before_sleep`: [optional] - Sleep time in seconds to sleep before action
- `after_sleep`: [optional] - Sleep time in seconds to sleep after action

## To do

- [ ] tests
- [ ] Better document `find_text` - write about how to provide Google Api Key
- [ ] conditional statements support
- [ ] loops support
- [ ] variables support - so one could run workflow many times with different data (e.g. for different accounts)
- [ ] UI - probably webUI
