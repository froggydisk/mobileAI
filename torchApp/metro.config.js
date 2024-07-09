const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('metro-config').MetroConfig}
 */
const defaultConfig = getDefaultConfig(__dirname);
const defaultAssetExts =
  require('metro-config/src/defaults/defaults').assetExts;

const config = {
  resolver: {
    assetExts: [...defaultAssetExts, 'ptl'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
