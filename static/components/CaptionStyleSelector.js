const React = window.React;
const { useState } = React;

const CaptionStyleSelector = () => {
    const [selectedStyle, setSelectedStyle] = useState('karaoke');

    const captionStyles = [
        {
            name: 'Typewriter',
            description: 'Text appears letter by letter',
            value: 'typewriter'
        },
        {
            name: 'Bounce In',
            description: 'Text bounces into view',
            value: 'bounce'
        },
        {
            name: 'Glow Effect',
            description: 'Words glow when spoken',
            value: 'glow_effect'
        },
        {
            name: 'Submagic',
            description: 'Words scale and highlight',
            value: 'submagic'
        },
        {
            name: 'Gradient Text',
            description: 'Colorful gradient effects',
            value: 'gradient'
        },
        {
            name: 'Karaoke',
            description: 'Classic karaoke highlighting',
            value: 'karaoke'
        }
    ];

    return React.createElement("div", { className: "space-y-4" },
        React.createElement("h3", { className: "text-lg font-medium" }, "Choose Caption Style"),
        React.createElement("div", { className: "grid grid-cols-2 md:grid-cols-3 gap-4" },
            captionStyles.map(style => 
                React.createElement("button", {
                    key: style.value,
                    onClick: () => {
                        setSelectedStyle(style.value);
                        document.getElementById('captionStyle').value = style.value;
                    },
                    className: `p-4 rounded-lg border transition-all ${
                        selectedStyle === style.value
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
            name: "captionStyle",
            id: "captionStyle",
            value: selectedStyle
        })
    );
};

export { CaptionStyleSelector as default };
