const React = window.React;
const { useState } = React;

const StyleSelector = () => {
    const [selectedStyle, setSelectedStyle] = useState('Default');

    const imageStyles = [
        {
            name: 'Default',
            description: 'Standard balanced style'
        },
        {
            name: 'Realistic',
            description: 'True-to-life representation'
        },
        {
            name: 'Cinematic',
            description: 'Movie-like dramatic scenes'
        },
        {
            name: 'Photographic',
            description: 'Professional photography look'
        },
        {
            name: '3D Model',
            description: '3D rendered visualization'
        },
        {
            name: 'Anime',
            description: 'Japanese animation style'
        },
        {
            name: 'Digital Art',
            description: 'Modern digital illustration'
        },
        {
            name: 'Dark',
            description: 'Moody and atmospheric'
        }
    ];

    return React.createElement("div", { className: "space-y-4" },
        React.createElement("h3", { className: "text-lg font-medium" }, "Choose Image Style"),
        React.createElement("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4" },
            imageStyles.map(style => 
                React.createElement("button", {
                    key: style.name,
                    onClick: () => {
                        setSelectedStyle(style.name);
                        document.getElementById('imageStyle').value = style.name;
                    },
                    className: `p-4 rounded-lg border transition-all ${
                        selectedStyle === style.name
                            ? 'border-blue-500 bg-blue-50 shadow-md'
                            : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                    }`
                },
                    React.createElement("div", { className: "text-center" },
                        React.createElement("div", { className: "font-medium" }, style.name),
                        React.createElement("div", { className: "text-sm text-gray-500" }, style.description)
                    )
                )
            )
        ),
        React.createElement("input", {
            type: "hidden",
            name: "imageStyle",
            id: "imageStyle",
            value: selectedStyle
        })
    );
};

export default StyleSelector;